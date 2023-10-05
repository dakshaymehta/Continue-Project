#this is going to be book or movie recommender system using techniques like collaborative filtering or content-based filtering.

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split


data = {'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        'movie_id': [1, 2, 1, 3, 2, 3, 1, 4, 4, 5],
        'rating': [5, 4, 4, 2, 3, 5, 4, 5, 3, 2]}

df = pd.DataFrame(data)

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2)


model = SVD()


model.fit(trainset)

predictions = model.test(testset)

rmse = accuracy.rmse(predictions)
print(f'RMSE: {rmse}')

# for each user 
def get_top_n(predictions, n=3):
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid not in top_n:
            top_n[uid] = []
        top_n[uid].append((iid, est))



    # sorting it for each user here
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

# Get top 3 recommendations for each user given after this point 

top_n = get_top_n(predictions, n=3)


for uid, user_ratings in top_n.items():
    print(f"Recommended items for user {uid}: {', '.join([str(iid) for iid, _ in user_ratings])}")
