# src/models/collaborative_filtering.py

import numpy as np
import pandas as pd

def svd_recommender(ratings, user_id, books, top_n=10, k=50):
    user_item_matrix = ratings.pivot(index='User-ID', columns='ISBN', values='Book-Rating')
    user_ratings_mean = user_item_matrix.mean(axis=1)
    R_demeaned = user_item_matrix.sub(user_ratings_mean, axis=0)

    U, sigma, Vt = np.linalg.svd(R_demeaned.fillna(0), full_matrices=False)
    sigma = np.diag(sigma)

    all_user_predicted_ratings = np.dot(np.dot(U[:, :k], sigma[:k, :k]), Vt[:k, :]) + user_ratings_mean.values[:, np.newaxis]
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns=user_item_matrix.columns, index=user_item_matrix.index)

    sorted_user_predictions = preds_df.loc[user_id].sort_values(ascending=False)
    user_data = ratings[ratings['User-ID'] == user_id]
    read_books = user_data['ISBN'].values
    recommendations = sorted_user_predictions[~sorted_user_predictions.index.isin(read_books)].head(top_n)

    return books[books['ISBN'].isin(recommendations.index)]
