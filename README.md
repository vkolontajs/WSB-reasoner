# WSB-reasoner 🚀

[Check out projects' Streamlit visuals](https://share.streamlit.io/vkolontajs/wsb-reasoner/main/streamlit_visual.py)

r/wallstreetbets comments reasoner for tickers using VADER scores.

The algorithm consists of two parts. 

1. The 'schedule_upload.py' runs every 30 minutes and loads the most available comments which then gets pre-processed, analyzed, and uploaded to the user's database. Tickers list for searching is from AlphaVantage. Analysis performed using VADER scoring from nltk package. Scores and comments uploaded to users' database (MongoDB).
2. The 'streamlit_visuals.py' is a Streamlit python file that is used for creating dashboards.

## Installation

Clone the project and run miniconda/anaconda.

Set credentials in settings.py for MongoDB and AlphaVantage.

```bash
conda env create -f conda_requirements.yml
conda activate WSB-reasoner

# For data loading and VADER analysis
python scheduled_upload.py

# For visualization of data
streamlit run streamlit_visuals.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
