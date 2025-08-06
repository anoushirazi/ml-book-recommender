# src/models/popularity_based.py

def popularity_recommender(ratings, books, top_n=10):
    pop_scores = ratings.groupby('ISBN')['Book-Rating'].count().sort_values(ascending=False).head(top_n)
    return books[books['ISBN'].isin(pop_scores.index)]
