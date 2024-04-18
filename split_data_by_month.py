'''
This file splits a single year of data into files for each month.

helpful for two reasons: 1) entire year csv slows my laptop on manual inspection
and 2) will be easier to pass individual months through OpenAI/Eikon APIs than working
with the entire year of data

Reference:
For help with learning how to manipulate pandas dataframes, I have asked GPT for examples.
I have also linked documentation that I found to be helpful for learning how to use libraries/APIs throughout the code.
'''

import pandas as pd

df = pd.read_csv("articles_2023.csv")


# see docs: https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html
# since the news API response dates are not always provided in a standardized form,
# needed to use format='mixed' argument in pd.to_datetime to successfully convert them
df['publishedAt'] = pd.to_datetime(df['publishedAt'], utc=True, format='mixed')
df['month'] = df['publishedAt'].dt.month

# see docs: https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html
# simple way to group by month and split into separate files
grouped = df.groupby('month')
for month, group in grouped:
    # see docs: https://www.geeksforgeeks.org/how-to-add-leading-zeros-to-a-number-in-python/
    # to add leading 0 in f string
    group.to_csv(f"raw_articles_by_month/articles_2023_month_{month:02d}.csv", index=False)