# src/data/data_preprocessor.py

def filter_top_users_books(ratings, top_n_users=1000, top_n_books=1000):
    top_users = ratings['User-ID'].value_counts().head(top_n_users).index
    top_books = ratings['ISBN'].value_counts().head(top_n_books).index
    ratings_filtered = ratings[ratings['User-ID'].isin(top_users) & ratings['ISBN'].isin(top_books)]
    return ratings_filtered
