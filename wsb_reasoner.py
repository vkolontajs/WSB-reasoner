# Python standard packages
import os
import re
import shutil
from datetime import datetime, timedelta

import pandas as pd
# Data loading and uploading packages
import pymongo
import requests
# NLTK tools for sentiment analysis
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pmaw import PushshiftAPI

# Settings files
import settings


def update_tickers_list_in_db():
    """
    Procedure retrieves tickers from alphavantage, filters for black listed keys and loads filtered tickers into user
    db.
    """

    # Get csv response
    base_url = 'https://www.alphavantage.co/query?'
    response = requests.get(
        base_url,
        params={
            'function': 'LISTING_STATUS',
            'state': 'active',
            'apikey': settings.alpha_api_key
        }
    )

    # Get list of active only tickers
    data = [row.strip().split(',') for row in response.text.split('\n')]
    df = pd.DataFrame(data[1:-1], columns=data[0])
    df = df[df.status == 'Active']
    tickers_list = df.symbol.to_list()

    # Filter out blacklisted items
    tickers_list = [ticker for ticker in tickers_list if ticker not in settings.black_listed_keys]

    # Update tickers in database
    db_handler = pymongo.MongoClient(settings.mongodb_connection_string)[settings.wall_db_name][settings.tickers_keys]
    db_handler.delete_many({})
    db_handler.insert_one({'tickers': tickers_list})


def get_tickers_list_from_db():
    """
    Function returns Timestamp with the date and time of the latest mention in db.
    :return: pd.Timestamp
    """

    db_handler = pymongo.MongoClient(settings.mongodb_connection_string)[settings.wall_db_name][settings.tickers_keys]
    tickers_list = db_handler.find({})

    return tickers_list[0]['tickers']


def get_comments_from_wallstreetbets(before, after):
    """
    Functions returns comments dataframe in particular horizon.
    :param before: pd.Timestamp()
    :param after: pd.Timestamp()
    :return: pd.DataFrame()
    """

    # Get cpu counts to specify maximum cores on VM available
    max_threads = os.cpu_count() * 5

    # Scrap comments from wallstreetbets
    api = PushshiftAPI()
    subreddit = "wallstreetbets"
    comments = api.search_comments(
        # PMAW parameters
        mem_safe=True,
        num_workers=max_threads,
        # Pushift.io parameters
        subreddit=subreddit,
        after=int(after.timestamp()),
        before=int(before.timestamp())
    )

    # Clean dataframe with comments
    comments_df = pd.DataFrame(comments)
    if not comments_df.empty:
        comments_df = comments_df[['id', 'author', 'body', 'created_utc']].drop_duplicates()
        comments_df.created_utc = pd.to_datetime(comments_df.created_utc, unit='s')
        comments_df = comments_df[~comments_df.body.isin(['[removed]', '[deleted]'])]
        comments_df = comments_df.sort_values('created_utc').reset_index(drop=True)

    return comments_df


def get_mentions_and_vader_scores_from_comments(comments_df, tickers_list):
    """
    Function goes over each comments, filters out tickers and applies vader scores to each comment.
    :param comments_df: pd.DataFrame()
    :param tickers_list: list
    :return: pd.DataFrame()
    """
    # Return empty if no comments
    if comments_df.empty:
        return comments_df

    # Filter out tickers in comments
    def find_tickers_in_comment(body):

        tickers = []

        for word in body.split():
            word = re.sub('^[^a-zA-Z]*|[^a-zA-Z]*$', '', word)
            if word.isupper() and len(word) <= 5 and word in tickers_list: tickers.append(word)

        # Return none if empty list
        if not tickers:
            return None
        else:
            tickers = pd.unique(tickers).tolist()
            if len(tickers) <= settings.max_comment_mentions:
                return tickers
            else:
                return None

    # Filter out comments w/out mentions
    comments_df['tickers'] = comments_df.body.apply(lambda body: find_tickers_in_comment(body))
    comments_df = comments_df[~pd.isna(comments_df.tickers)].reset_index(drop=True)

    # Prepare vader scores
    if comments_df.empty:
        return comments_df
    else:
        # Get vader scores
        vader = SentimentIntensityAnalyzer()
        comments_df['vader_scores'] = comments_df.body.apply(lambda x: vader.polarity_scores(x))
        comments_df[['neg', 'neu', 'pos', 'compound']] = comments_df.vader_scores.apply(pd.Series)

        # Pretify comments
        comments_df = comments_df.drop(columns=['vader_scores']).rename(columns={'body': 'comment'})

        return comments_df


def clean_db(before=None):
    """
    Procedure cleans db from all data if before datetime is not specified, or cleans mentions before the datetime.
    :param before: pd.Timestamp()
    """
    db_handler = pymongo.MongoClient(settings.mongodb_connection_string)[settings.wall_db_name]

    if before is None:
        db_handler[settings.comments_coll].delete_many({})
        db_handler[settings.tickers_keys].delete_many({})
    else:
        db_handler[settings.comments_coll].delete_many({'created_utc': {'$lte': before}})


def upload_comments_and_scores_to_db(comments):
    """
    Procedure loads dataframe to db.
    :param comments: pd.DataFrame()
    """
    db_handler = pymongo.MongoClient(settings.mongodb_connection_string)[settings.wall_db_name][settings.comments_coll]
    db_handler.insert_many(comments.to_dict('records'))


def get_comments_and_scores_from_db():
    """
    Function gets all comments with vader scores from db.
    :return: pd.DataFrame()
    """
    db_handler = pymongo.MongoClient(settings.mongodb_connection_string)[settings.wall_db_name][settings.comments_coll]
    comments = pd.DataFrame(db_handler.find({}, {'_id': 0}))

    if not comments.empty:
        comments = comments.astype({'comment': 'str', 'id': 'str'})

    return comments


def get_datetime_of_the_last_mention_in_db():
    """
    Function returns timestamp with the date of the last mention in db. If there is none, returns None.
    :return: pd.Timestamp or None
    """
    db_handler = pymongo.MongoClient(settings.mongodb_connection_string)[settings.wall_db_name][settings.comments_coll]
    max_date = db_handler.find_one({}, sort=[("created_utc", pymongo.DESCENDING)])

    if max_date is not None:
        max_date = pd.to_datetime(max_date['created_utc'])

    return max_date


def update_last_upload_time(upload_time):
    """
    Procedure loads upload last upload time to db
    :return: None
    """
    db_handler = pymongo.MongoClient(settings.mongodb_connection_string)[settings.wall_db_name][
        settings.last_upload_time]
    db_handler.delete_many({})
    db_handler.insert_one({'upload_time': upload_time})


def get_last_upload_time():
    """
    Procedure loads upload last upload time to db
    :return: None
    """
    db_handler = pymongo.MongoClient(settings.mongodb_connection_string)[settings.wall_db_name][
        settings.last_upload_time]
    date = pd.to_datetime(db_handler.find_one({})['upload_time'])
    return date


def complete_flow(reset=False):
    """
    Procedure is a full process of data process for tickers sentiment analysis and db uploading. If you are
    restarting db then put reset variable to True.
    :param reset: Boolean
    """

    nltk.download('vader_lexicon')

    # Clean database
    if reset:
        clean_db()

    # Process flow
    current_utc = pd.to_datetime(datetime.utcnow())
    cut_off_utc = current_utc - timedelta(hours=settings.cut_off_hours)

    # Update tickers with data from Alphavantage
    update_tickers_list_in_db()

    # Start process of loading comments and application of vader scores
    last_utc = get_datetime_of_the_last_mention_in_db()
    if last_utc is None:
        last_utc = cut_off_utc

    # Load data, filter out tickers and apply vader scores
    for after in pd.date_range(last_utc, current_utc + timedelta(hours=1), freq='1H'):

        # Clean cache
        shutil.rmtree(r'cache', ignore_errors=True)

        before = after + timedelta(hours=1)

        print('\nLoading data for this horizon: ')
        print('After: ' + after.strftime('%Y-%m-%d %H:%M:%S'))
        print('Before: ' + before.strftime('%Y-%m-%d %H:%M:%S') + '\n')

        # Get comments mentions and scores
        comments_df = get_mentions_and_vader_scores_from_comments(
            get_comments_from_wallstreetbets(
                after=after,
                before=before
            ),
            get_tickers_list_from_db()
        )

        if not comments_df.empty:
            upload_comments_and_scores_to_db(comments_df)

    # Clean old data before cut-off datetime
    clean_db(before=cut_off_utc)

    # Update last upload time
    update_last_upload_time(pd.to_datetime(datetime.utcnow()))

    # Clean cache
    shutil.rmtree(r'cache', ignore_errors=True)
