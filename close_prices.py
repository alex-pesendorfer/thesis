'''
This file reads a csv of the stock-tagged articles from GPT, which each have a RIC tag.
By making calls to the Eikon API, we retrieve the close prices for days t-2, t, t+1 for 
each article where day t is the publishing day. When markets are closed on the publishing day,
we look forward for the next trading day. We also retrieve close prices for SPY for the same days.

Note that the Refinitiv Workspace application must be running in the background to retrieve close prices
through the Eikon API.

Reference:
For help with learning how to manipulate pandas dataframes and how to use datetime, I have asked GPT for examples.
I have also linked documentation that I found to be helpful for learning how to use libraries/APIs throughout the code.
'''

import pandas as pd
import numpy as np
import eikon as ek
import os
from datetime import timedelta, datetime
import time

# see docs: https://developers.lseg.com/content/dam/devportal/api-families/eikon/eikon-data-api/documentation/eikon_data_api_for_python_v1.pdf
# note, in order to run the Eikon code, one must have the Refinitiv Workplace app running
# and have requested an API key via Refinitiv (university access)
ek.set_app_key(os.environ["EIKON_API_KEY"])

def get_close_prices(ric, publish_date):
    # the below conversions are necessary because we need to truncate the additional time
    # information in the timestamps before passing them to the Eikon API

    # convert string publish_date to datetime
    publish_date = pd.to_datetime(publish_date, utc=True, format='mixed')
    # convert datetime to string in simplified year-month-day format
    publish_date = publish_date.strftime('%Y-%m-%d')
    # convert back to datetime with simplified year-month-day format
    publish_date = datetime.strptime(publish_date, "%Y-%m-%d")
    
    # pad start and end dates in case markets are closed on the day of publishing
    # and we need to look earlier/later for the training data
    # see docs: https://www.geeksforgeeks.org/how-to-add-days-to-a-date-in-python/
    start_date = publish_date - timedelta(days=10)
    end_date = publish_date + timedelta(days=10)

    try:
        # convert start_date, end_date to string format required by Eikon API
        data = ek.get_timeseries(ric, 
                                 start_date=start_date.strftime('%Y-%m-%d'), 
                                 end_date=end_date.strftime('%Y-%m-%d'), 
                                 fields='CLOSE')
        
        print(data.index)
        
        # data retrievied is a dataframe with datetime index and column CLOSE
        # if publish_date is not in the dataframe, we will find the next valid date
        # and act as though that is the publishing date (note in cases where this occurs,
        # it only further delays our ability to act on news, so it provides no advantage)
        while publish_date not in data.index and publish_date <= end_date:
            publish_date += timedelta(days=1)

        # see docs: https://pandas.pydata.org/docs/reference/api/pandas.Index.get_loc.html
        # find numerical index for the publish_date
        t = data.index.get_loc(publish_date)
        # retrieve the datetimes corresponding to t-2, t, and t+1 from the dataframe
        dates = [data.index[t-2], data.index[t], data.index[t+1]]
        # retrieve the dataframe of close prices for these dates
        close_prices = data.loc[dates]
        return close_prices
    
    except Exception as e:
        print("Exception:", e)
        return None

# print(get_close_prices('PETR4.SA', '2023-01-01 19:59:59.924000+00:00'))
df = pd.read_csv('stock_tagged_final_data_by_month/articles_2023_month_12.csv')

# see docs: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html
# drop rows with no valid RIC
df.dropna(subset=['RIC'], inplace=True)
df = df[df['RIC'] != 'N/A']

# some of the RIC labels from GPT will be invalid, which will cause errors
# in the calls to the Eikon API. thus, we keep track of the rows that produce errors 
# and drop them at the end
df['error'] = False

# instantiate empty columns that will be used for the close prices of article-specific stocks and SPY
new_columns = ['t-2', 't', 't+1', 'SPY_t-2', 'SPY_t', 'SPY_t+1']
for col in new_columns:
    df[col] = np.nan

try:
    # first retrieve SPY close prices for the year 2023 with some padding at start and end
    spy_data = ek.get_timeseries('SPY.P', start_date='2022-12-21', end_date='2024-01-10', fields='CLOSE')
    time.sleep(2) # avoid rate limit

    # important to iterate row by row rather than making concurrent calls
    # because Eikon API rate limits are quite strict
    for index, row in df.iterrows():
        ric = row['RIC']
        publish_date = row['publishedAt']
        close_prices_df = get_close_prices(ric, publish_date)
        time.sleep(2)
        
        # error with this row, add flag for later deletion
        if close_prices_df is None:
            # see docs: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.at.html
            df.at[index, 'error'] = True
            continue

        # set close prices in dataframe
        df.loc[index, ['t-2', 't', 't+1']] = close_prices_df['CLOSE'].values
        
        dates = [close_prices_df.index[i] for i in range(0, 3)]

        # get SPY close prices for same days (for use as market factor in regressions
        # and for comparison in portfolio performance)

        # possible that SPY did not trade on the days that the stock traded
        # since we have articles about stocks trading on international markets
        df.at[index, 'SPY_t-2'] = spy_data.at[dates[0], 'CLOSE'] if dates[0] in spy_data.index else np.nan
        df.at[index, 'SPY_t'] = spy_data.at[dates[1], 'CLOSE'] if dates[1] in spy_data.index else np.nan
        df.at[index, 'SPY_t+1'] = spy_data.at[dates[2], 'CLOSE'] if dates[2] in spy_data.index else np.nan

        # if index > 50: # for testing, interrupt early
        #     break
except Exception as e:
    print("Exception:", e)
finally:        
    # drop the rows that had an error (due to invalid RIC provided by GPT in labeling process)
    df = df[~df['error']]
    # drop the error column before writing to csv
    df.drop(columns=['error'], inplace=True)
    df.to_csv('new_data/reg_data_with_close/articles_2023_month_12.csv', index=False)