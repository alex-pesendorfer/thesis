'''
This file makes calls to OpenAI chat completions API to assign RIC stock tags and sentiment labels
to each article in the data.

Reference:
For help with learning how to manipulate pandas dataframes, I have asked GPT for examples.
I have also linked documentation that I found to be helpful for learning how to use libraries/APIs throughout the code.
'''

import pandas as pd
import json
from openai import OpenAI
client = OpenAI()

df = pd.read_csv("cleaned_categorized_embedded_articles_by_month/articles_2023_month_12_with_categories.csv")
df['RIC'], df['sentiment'] = "", ""

# see docs: https://platform.openai.com/docs/guides/text-generation/chat-completions-api for OpenAI chat completions
# also: https://platform.openai.com/docs/api-reference/chat/object?lang=python
# this function tags a row, which corresponds to a news article, with the corresponding company RIC and a sentiment label
# if no publicly-listed company is found or there is an error, we default to "N/A" for the RIC and sentiment label
def stock_tag(row):
    content_type = "title" if (pd.isnull(row['description']) or row['description'] == "") else "description"
    content = row[content_type]
    if content and content != "":
        print(content)
        messages = [
            {'role': 'system', "content": 'You will be given the ' + content_type + ' of a financial news article. The ' + content_type + ' may reference one or more companies listed on a major exchange. If so, output two things: 1) the RIC (Refinitiv Instrument Code) of the primary company on a major exchange and 2) exactly one of: "positive", "neutral", or "negative" based on whether the ' + content_type + ' contains information that is positive, neutral, or negative in terms of possible stock price moves of the identified company. Output format: {"RIC": RIC, "sentiment": sentiment}. If the news is positive/negative for some other party or the public, do not let that impact your classification. Focus entirely on the effect on the company stock price. If there is not a clear positive or negative direction in the news with respect to the interests of the company, classify it as neutral. If there is no relevant company or the company is not listed on a major exchange, output "N/A". Some of the common RIC endings are: ".N", ".OQ", ".T", ".L", ".HK", ".SS", ".SZ", ".F", ".PA", ".AS", ".MI", ".BR", ".TO", ".AX", ".BO", ".NS", ".BM", ".KS", ".TW", ".SI", ".ST", ".MI", ".MM", each of which corresponds to a different exchange. Finally, be sure to output this information for only one company, even if the description references multiple companies: select the most important.'},
            {'role': 'user', 'content': content_type.capitalize() + ': ' + content},
        ]
        try:
            response = client.chat.completions.create(
                model='gpt-4-1106-preview',
                messages=messages,
                temperature=0.0,
            )
            string_response = response.choices[0].message.content
            print(string_response + "\n")

            if 'N/A' in string_response:
                return "N/A", "N/A"
            else:
                result = json.loads(string_response)
                return result["RIC"], result["sentiment"]
        except Exception as e:
            print("Exception:", e)
            print("df row:", row)
            return "N/A", "N/A"
    return "N/A", "N/A"

# see docs: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iterrows.html
# though often preferable to apply operations in more efficient ways, I opt for row-by-row
# manipulation here since I don't want to exceed API rate limits and I also want to save the 
# current progress to a csv if there are any errors that interrupt the code
for index, row in df.iterrows():
    df.at[index, 'RIC'], df.at[index, 'sentiment'] = stock_tag(row)
    # save periodically in case bugs occur
    if index > 0 and index % 500 == 0:
        df.to_csv("stock_tagged_final_data_by_month/articles_2023_month_12.csv", index=False)

df.to_csv("stock_tagged_final_data_by_month/articles_2023_month_12.csv", index=False)