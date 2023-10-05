#testing 

import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

nltk.download('vader_lexicon')

# preparing the sample data for the sentimental analysis

data = {'Customer_Reviews': ['The product is excellent!', 
                             'Do not buy this product. It is terrible.', 
                             'It is okay, not the best.',
                             'Amazing quality!',
                             'I would use this for sure', 
                             'Going to buy it again', 
                             'Cant wait to recomend this to my freinds', 
                             'Loved it',
                             'Waste of money.', 'I would not buy this thing', 'Worst thing on the market']}

df = pd.DataFrame(data)

sia = SentimentIntensityAnalyzer()


# Apply the polarity_scores method of SentimentIntensityAnalyzer to each customer review
# Apply sentiment analysis for each review
df['sentiment_scores'] = df['Customer_Reviews'].apply(lambda review: sia.polarity_scores(review))

# Classify reviews as positive, neutral or negative based on the compound score
df['sentiment'] = df['sentiment_scores'].apply(lambda score_dict: 'positive' if score_dict['compound'] > 0 
                                               else 'negative' if score_dict['compound'] < 0 else 'neutral')

print(df)

plt.bar(df['sentiment'].value_counts().index, df['sentiment'].value_counts().values)
plt.show()
