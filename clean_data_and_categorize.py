'''
This code removes duplicates articles from the set by looking for those that have
cosine similarity greater than 0.97 (a threshold somewhat arbitrarily chosen by inspection).

Then, we perform zero-shot classification of each article into one of eight manually-selected
news categories by maximizing cosine similarity with the embedded category labels.

Reference:
For help with learning how to manipulate pandas dataframes and how to use scikit-learn, I have asked GPT for examples.
I have also linked documentation that I found to be helpful for learning how to use libraries/APIs throughout the code.
'''

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

client = OpenAI()

# see docs: https://platform.openai.com/docs/guides/embeddings/use-cases
# and https://platform.openai.com/docs/api-reference/embeddings/create
# and https://platform.openai.com/docs/guides/embeddings/what-are-embeddings 
# for zero-shot clasification with embeddings, which we use for news category classification
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


# manually-selected categories
news_categories = ["Supply or Demand Effect", "Earnings Expectations or Earnings Reports", 
              "Merger or Acquisition", "Legal and Regulatory", "Fines Incurred", 
              "Personnel/Management Changes", "New Technology and Innovation", "Product Launch"]

# embed the category labels
news_cat_embeddings = [get_embedding(news_category) for news_category in news_categories]

df = pd.read_csv("embedded_articles_by_month/articles_2023_month_12_with_embeddings.csv")

# retrieve embeddings from df (we stored them across columns in embeddings.py since had issues with 
# saving string-representation of arrays in csv)
embedding_cols = ['embedding_' + str(i) for i in range(1536)]

# see docs: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
# scikit-learn library function to compute cosine similarity matrix given a list of vectors 
# (by computing pairwise similarities for all vectors in the list)
sim_matrix = cosine_similarity(df[embedding_cols].values)

# remove subsequent articles that exceed similarity threshold with earlier article
# this threshold was somewhat arbitrarily chosen by testing manually to see when 
# clearly-duplicated articles were successfully removed
cutoff = 0.97
duplicates = set()
for i in range(len(sim_matrix)):
    for j in range(i + 1, len(sim_matrix)):
        if sim_matrix[i][j] > cutoff:
            duplicates.add(j)

# see docs: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html
# drop the rows identified as duplicates from the df and reset indices
df = df.drop(index=duplicates).reset_index(drop=True)

# assign categories by highest cosine similarity (see OpenAI zero-shot classification docs above)
# similarity matrix is of dimension (num embeddings) x (num categories)
# so to assign categories, we assign each article (row) the category with index
# equal to argmax over the columns (see below stackoverflow)
# https://stackoverflow.com/questions/71376238/how-to-find-most-optimal-match-between-data-points-given-the-similarity-matrix
cat_sim_matrix = cosine_similarity(df[embedding_cols].values, news_cat_embeddings)
df['category'] = [news_categories[i] for i in np.argmax(cat_sim_matrix, axis=1)]

df.to_csv("cleaned_categorized_embedded_articles_by_month/articles_2023_month_12_with_categories.csv", index=False)
