from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

finviz_url = 'https://finviz.com/quote.ashx?t='
tickers = ['AMZN', 'AMD', 'META', 'GOOG', 'NVDA', 'VOO', 'SPY']
news_tables = {}

for ticker in tickers:
    # make the actual url for each ticker to look at
    url = finviz_url + ticker

    req = Request(url=url, headers={'user-agent': 'my-app'})
    response = urlopen(req)

    html = BeautifulSoup(response, 'html')
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table



parsed_data = []
#try to only get ~20 rows per ticker
for ticker, news_table in news_tables.items():
    for row in news_table.findAll('tr'):
        title = row.a.get_text()
        date_data = row.td.text.split(' ')

        if len(date_data) == 21:
            time = date_data[12]
        else:
            date = date_data[12]
            time = date_data[13]

        time = time.rstrip("\r\n")
        parsed_data.append([ticker, date, time, title])


df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])

vader = SentimentIntensityAnalyzer()

f = lambda title: vader.polarity_scores(title)['compound']
df['compound'] = df['title'].apply(f)
df['date'] = pd.to_datetime(df['date'], format='%b-%d-%y').dt.date

# Define the cutoff date for the last 5 days
cutoff_date = datetime.now().date() - timedelta(days=5)

# Filter the DataFrame to keep only dates within the last 5 days
df = df[df['date'] >= cutoff_date]

plt.figure(figsize=(10,8))

mean_df = df.groupby(['ticker', 'date']).mean(numeric_only=True)
mean_df = mean_df.unstack()
mean_df = mean_df.xs('compound', axis="columns").transpose()
mean_df.plot(kind='bar')
plt.show()

