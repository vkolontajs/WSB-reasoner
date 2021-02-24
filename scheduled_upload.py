from wsb_reasoner import complete_flow
import schedule

if __name__ == '__main__':
    print('The process has started: ')
    complete_flow()

    schedule.every(30).minutes.do(complete_flow)

    while True:
        # Checks whether a scheduled task is pending to run or not
        schedule.run_pending()