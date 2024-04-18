'''
This code retrieves Reuters and Bloomberg articles for the year 2023 through the newsfilter.io API.

Reference:
For help with learning how to use csv library, I have asked GPT for examples.
I have also linked documentation that I found to be helpful for learning how to use libraries/APIs throughout the code.
'''

import csv
import datetime
import os
import requests
import time

API_KEY = os.environ["NEWS_FILTER_API_KEY"]
API_ENDPOINT = "https://api.newsfilter.io/public/actions?token={}".format(API_KEY)

# see docs: https://developers.newsfilter.io/docs/news-query-api-example-python.html
# this function retrieves 25 articles starting at the index from_param for the specified date
# due to rate limits, we retrieve small chunks and leverage the fact that we can choose the starting index
def query(date, from_param):
    queryString = "(source.id:reuters OR source.id:bloomberg) AND publishedAt:[" + str(date) + " TO " + str(date) + "]"
    payload = {
        "type": "filterArticles",
        "queryString": queryString,
        "from": from_param,
        "size": 25
    }

    response = requests.post(API_ENDPOINT, json=payload)
    return response.json()

l_date, r_date = datetime.date(2023, 1, 1), datetime.date(2023, 12, 31)
news_articles = []

# below, response['total']['value'] contains the total number of articles that match the search query

# with free tier limitations, can only make small requests, so we increment the from_param value
# to keep track of our current position in the total count of matching results for the particular day

# when this from_param meets (or exceeds, though should not occur) the total matches, we break the loop,
# reset the from_param to 0, and increment the left date pointer to move to the next day of articles
while l_date <= r_date:
    print(l_date)
    from_param = 0
    while True:
        response = query(l_date.strftime("%Y-%m-%d"), from_param)
        if response and response['articles']:
            total_count = response['total']['value']
            news_articles.extend(response['articles'])
            from_param += len(response['articles'])
            time.sleep(1)  # avoid rate limit
            if from_param >= total_count:
                break
        else:
            break
    l_date += datetime.timedelta(days=1)

# see docs: https://docs.python.org/3/library/csv.html
with open('articles_2023.csv', 'w', newline='') as csvfile:
    fieldnames = list(news_articles[0].keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(news_articles)