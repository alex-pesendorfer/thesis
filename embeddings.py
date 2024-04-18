'''
This code takes the csv of articles retrieved via newsfilter.io API and embeds the content of each article.
We opt to embed the descriptions whenever available since they tend to be longer and encode more information than
the titles. Then, we save the embeddings into the csv such that each component has its own column "embedding_i" for
integer i in [0, 1535]. I found this to be easier to work with than saving the embedding as a string in a single column
of the csv.

Reference:
For help with learning how to manipulate pandas dataframes, I have asked GPT for examples.
I have also linked documentation that I found to be helpful for learning how to use libraries/APIs throughout the code.
'''

import pandas as pd
import time
from openai import OpenAI

client = OpenAI()

# see docs: https://platform.openai.com/docs/api-reference/embeddings/create
def get_embedding(texts):
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [item.embedding for item in response.data]

df = pd.read_csv("raw_articles_by_month/articles_2023_month_12.csv")

content_list = []
for index, row in df.iterrows():
    content_type = "title" if (pd.isnull(row['description']) or row['description'] == "") else "description"
    content = row[content_type]
    content_list.append(content)

all_embeddings = []
# an input array to OpenAI embeddings API must have less than 2048 dimensions
# we choose to make calls in batches of 1000
# see docs: https://www.geeksforgeeks.org/break-list-chunks-size-n-python/
chunk_size = 1000
for i in range(0, len(content_list), chunk_size):  
    texts = content_list[i:i + chunk_size]
    embeddings = get_embedding(texts)
    all_embeddings.extend(embeddings)
    time.sleep(1) # avoid rate limits

# create new df for embeddings (we will store each component in a new column, since storing
# the entire 1536 dimensional vector as a string representation of an array proved more challenging 
# to work with later on)
# see docs: https://www.geeksforgeeks.org/convert-a-numpy-array-to-pandas-dataframe-with-headers/
embedding_df = pd.DataFrame(all_embeddings, columns=['embedding_' + str(i) for i in range(len(all_embeddings[0]))])
df = pd.concat([df, embedding_df], axis=1)

df.to_csv("embedded_articles_by_month/articles_2023_month_12_with_embeddings.csv", index=False)