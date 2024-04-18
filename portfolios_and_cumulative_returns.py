'''
This file constructs long-short portfolios for each of the three strategies, plotting their cumulative returns
and printing performance metrics. It also includes SPY in the plot for reference.

Reference:
For help with learning how to manipulate pandas dataframes and how to use matplotlib/scikit-learn, I have asked GPT for examples.
I have also linked documentation that I found to be helpful for learning how to use libraries/APIs throughout the code.
'''

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def print_return_info(return_vec, cumulative_return):
    avg_daily_return = return_vec.mean()
    stdev_daily_return = return_vec.std()
    print("avg_daily_return:", avg_daily_return)
    print("stdev_daily_return:", stdev_daily_return)
    sharpe_ratio = (avg_daily_return / stdev_daily_return) * np.sqrt(252)
    print("Annualized Sharpe Ratio:", sharpe_ratio)
    cumulative_return = (1 + return_vec).prod()
    print("Cumulative Return:", cumulative_return)

# see docs: https://stackoverflow.com/questions/20906474/import-multiple-csv-files-into-pandas-and-concatenate-into-one-dataframe
# need to concatenate all data into one dataframe
all_files = [
    'reg_data_with_close/articles_2023_month_01.csv',
    'reg_data_with_close/articles_2023_month_02.csv',
    'reg_data_with_close/articles_2023_month_03.csv',
    'reg_data_with_close/articles_2023_month_04.csv',
    'reg_data_with_close/articles_2023_month_05.csv',
    'reg_data_with_close/articles_2023_month_06.csv',
    'reg_data_with_close/articles_2023_month_07.csv',
    'reg_data_with_close/articles_2023_month_08.csv',
    'reg_data_with_close/articles_2023_month_09.csv',
    'reg_data_with_close/articles_2023_month_10.csv',
    'reg_data_with_close/articles_2023_month_11.csv',
    'reg_data_with_close/articles_2023_month_12.csv',
]
df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)

# usual datetime conversion since news article publishing times are in mixed formats
df['timestamp'] = pd.to_datetime(df['publishedAt'], utc=True, format='mixed')
df['date'] = df['timestamp'].dt.date

# convert all timestamps to EST so we can drop those outside of market hours
df['timestamp_est'] = df['timestamp'].dt.tz_convert('EST')

# compute hour as float for each article
df['hour_est'] = df['timestamp_est'].dt.hour + (df['timestamp_est'].dt.minute / 60)

# # testing
# split_ric = df['RIC'].str.split('.')
# markets = [item[1] for item in split_ric]
# df['market'] = markets
# markets = set(markets)
# print(markets)

# take subset of articles that were published within market hours for their respective RIC's market
df = df[(df['RIC'].str.endswith(".N") & ((df['hour_est'] >= 9.5) & (df['hour_est'] <= 16))) | 
        (df['RIC'].str.endswith(".T") & ((df['hour_est'] >= 19) | (df['hour_est'] <= 2))) | 
        (df['RIC'].str.endswith(".OQ") & ((df['hour_est'] >= 9.5) & (df['hour_est'] <= 16))) | 
        (df['RIC'].str.endswith(".L") & ((df['hour_est'] >= 3) & (df['hour_est'] <= 11.5))) |
        (df['RIC'].str.endswith(".NS") & ((df['hour_est'] >= 22.75) | (df['hour_est'] <= 5))) |
        (df['RIC'].str.endswith(".PA") & ((df['hour_est'] >= 3) & (df['hour_est'] <= 11.5))) | 
        (df['RIC'].str.endswith(".HK") & ((df['hour_est'] >= 21.5) | (df['hour_est'] <= 4))) |
        (df['RIC'].str.endswith(".SS") & ((df['hour_est'] >= 21.5) | (df['hour_est'] <= 3))) |
        (df['RIC'].str.endswith(".SZ") & ((df['hour_est'] >= 21.5) | (df['hour_est'] <= 3))) |
        (df['RIC'].str.endswith(".F") & ((df['hour_est'] >= 3) & (df['hour_est'] <= 11.5))) |
        (df['RIC'].str.endswith(".AS") & ((df['hour_est'] >= 3) & (df['hour_est'] <= 11.5))) |
        (df['RIC'].str.endswith(".MI") & ((df['hour_est'] >= 3) & (df['hour_est'] <= 11.5))) |
        (df['RIC'].str.endswith(".BR") & ((df['hour_est'] >= 3) & (df['hour_est'] <= 11.5))) |
        (df['RIC'].str.endswith(".TO") & ((df['hour_est'] >= 9.5) & (df['hour_est'] <= 16))) |
        (df['RIC'].str.endswith(".AX") & ((df['hour_est'] >= 18) | (df['hour_est'] <= 0))) |
        (df['RIC'].str.endswith(".BO") & ((df['hour_est'] >= 22.75) | (df['hour_est'] <= 5))) |
        (df['RIC'].str.endswith(".KS") & ((df['hour_est'] >= 19) | (df['hour_est'] <= 1.5))) |
        (df['RIC'].str.endswith(".TW") & ((df['hour_est'] >= 20.5) | (df['hour_est'] <= 1))) |
        (df['RIC'].str.endswith(".SI") & ((df['hour_est'] >= 20.5) | (df['hour_est'] <= 4))) |
        (df['RIC'].str.endswith(".ST") & ((df['hour_est'] >= 3) & (df['hour_est'] <= 11.5)))]

# drop rows that are missing values for these close prices
df = df.dropna(subset=['t-2', 't', 't+1', 'SPY_t-2', 'SPY_t', 'SPY_t+1'])
df['train_returns'] = (df['t+1'] - df['t-2']) / df['t-2']
df['test_returns'] = (df['t+1'] - df['t']) / df['t']
df['sr_sign'] = np.where(df['train_returns'] > 0, 1, -1)
df['SPY_test_returns'] = (df['SPY_t+1'] - df['SPY_t']) / df['SPY_t']

# convert sentiment label to number for later GPT-L logistic regression
sentiment_dict = {'positive': 1, 'neutral': 0, 'negative': -1}
df['gpt_d_label'] = df['sentiment'].map(sentiment_dict)

# split half train/half test by dates
dates = df['date'].unique()
split_index = int(len(dates) * 0.5)
train_dates, test_dates = dates[:split_index], dates[split_index:]

# see docs: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.isin.html
# filter by train/test dates
train_df, test_df = df[df['date'].isin(train_dates)], df[df['date'].isin(test_dates)]

# to test on just a news-category subset
# test_df = test_df[test_df['category'] == 'Earnings Expectations or Earnings Reports']

embedding_cols = ['embedding_' + str(i) for i in range(1536)]
X_train = train_df[embedding_cols].values
X_test = test_df[embedding_cols].values
y_train = train_df['sr_sign'].values

# see docs: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
pca = PCA(n_components=50, random_state=42)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# Stock-Return-Based Logistic Regression Strategy

# see docs: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#
# class_weight param used to adjust weights to increase value placed on infrequent labels
sr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
sr_model.fit(X_train, y_train)

# see docs: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.predict_proba
# obtain positive-label probabilities for each article
test_df['sr_prob'] = sr_model.predict_proba(X_test)[:, 1]

# see docs: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
# create new df for the returns, indexed by the day from test_dates
sr_return_df = pd.DataFrame(index=test_dates, columns=['return'])
for date in test_dates:
    temp_df = test_df[test_df['date'] == date]

    # see docs: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html
    # sort by logistic regression positive-class probability from high to low
    temp_df = temp_df.sort_values(by='sr_prob', ascending=False)

    # retrieve top and bottom deciles (or at least one element is len is less than 10)
    top_decile = temp_df.head(max(1, int(len(temp_df) * 0.1)))
    bottom_decile = temp_df.tail(max(1, int(len(temp_df) * 0.1)))

    # compute mean of top/bottom decile returns
    long_returns = top_decile['test_returns'].mean() if not np.isnan(top_decile['test_returns'].mean()) else 0
    short_returns = bottom_decile['test_returns'].mean() if not np.isnan(bottom_decile['test_returns'].mean()) else 0
    daily_return = long_returns - short_returns # long-short
    # daily_return = long_returns # long-only
    # daily_return = - short_returns # short-only

    # save daily return in new df
    sr_return_df.at[date, 'return'] = daily_return

sr_cumulative_return_vec = (1 + sr_return_df['return']).cumprod()
sr_cumulative_return = (1 + sr_return_df['return'] ).prod()

print("SR Performance:")
print_return_info(sr_return_df['return'], sr_cumulative_return_vec)


# Direct GPT Label Strategy
gpt_d_return_df = pd.DataFrame(index=test_dates, columns=['return'])
for date in test_dates:
    temp_df = test_df[test_df['date'] == date]
    positive_news = temp_df[temp_df['sentiment'] == "positive"]
    negative_news = temp_df[temp_df['sentiment'] == "negative"]
    long_returns = positive_news['test_returns'].mean() if not np.isnan(positive_news['test_returns'].mean()) else 0
    short_returns = negative_news['test_returns'].mean() if not np.isnan(negative_news['test_returns'].mean()) else 0
    daily_return = long_returns - short_returns
    # daily_return = long_returns
    # daily_return = - short_returns
    gpt_d_return_df.at[date, 'return'] = daily_return

gpt_d_cumulative_return = (gpt_d_return_df['return'] + 1).prod()
gpt_d_cumulative_return_vec = (1 + gpt_d_return_df['return']).cumprod()

print("GPT_D Performance:")
print_return_info(gpt_d_return_df['return'], gpt_d_cumulative_return_vec)

# GPT-Based Logistic Regression Strategy

y_train = train_df['gpt_d_label'].values
gpt_l_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
gpt_l_model.fit(X_train, y_train)

predicted_probabilities = gpt_l_model.predict_proba(X_test)
print(gpt_l_model.classes_) # order of classes is -1, 0, 1
test_df['gpt_l_negative_prob'] = predicted_probabilities[:, 0]  # logistic negative-label probability
test_df['gpt_l_positive_prob'] = predicted_probabilities[:, 2]  # logistic positive-label probability


gpt_l_return_df = pd.DataFrame(index=test_dates, columns=['return'])
for date in test_dates:
    temp_df = test_df[test_df['date'] == date]
    # now deciles are over two different label probabilities (most positive and most negative)
    # see docs: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.nlargest.html
    # we use nlargest to retrieve top 10% positive and top 10% negative
    top_decile = temp_df.nlargest(max(1, int(len(temp_df) * 0.1)), 'gpt_l_positive_prob')
    bottom_decile = temp_df.nlargest(max(1, int(len(temp_df) * 0.1)), 'gpt_l_negative_prob')
    long_returns = top_decile['test_returns'].mean() if not np.isnan(top_decile['test_returns'].mean()) else 0
    short_returns = bottom_decile['test_returns'].mean() if not np.isnan(bottom_decile['test_returns'].mean()) else 0
    
    daily_return = long_returns - short_returns
    # daily_return = long_returns
    # daily_return = - short_returns

    gpt_l_return_df.at[date, 'return'] = daily_return
    # for comparison, save SPY daily return (all entries on same date should be same so taking mean doesn't impact anything)
    gpt_l_return_df.at[date, 'SPY_return'] = df.loc[df['date'] == date, 'SPY_test_returns'].mean()

gpt_l_cumulative_return_vec = (1 + gpt_l_return_df['return']).cumprod()
spy_cumulative_return_vec = (1 + gpt_l_return_df['SPY_return']).cumprod()

print("GPT_L Performance:")
print_return_info(gpt_l_return_df['return'], gpt_l_cumulative_return_vec)


plt.figure(figsize=(10, 6))
sr_cumulative_return_vec.plot(label='SR Long-Short Cumulative Return', color='blue')
gpt_l_cumulative_return_vec.plot(label='GPT-L Long-Short Cumulative Return', color='green')
gpt_d_cumulative_return_vec.plot(label='GPT-D Long-Short Cumulative Return', color='red')
spy_cumulative_return_vec.plot(label='SPY Cumulative Return', color='orange')
plt.title('Performance of Long-Short Equal Weighted Portfolios vs SPY')
# plt.title('Performance of Long-Short Equal Weighted Portfolios over the \nEarnings Expectations or Earnings Reports Category vs SPY')

plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.show()

# compute different portoflio daily stock return correlations
corr = pd.concat([sr_return_df['return'], gpt_d_return_df['return'], gpt_l_return_df['return'], gpt_l_return_df['SPY_return']], axis=1).corr()
print(corr)
