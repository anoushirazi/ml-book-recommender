# src/data/data_cleaner.py

import numpy as np

def clean_books(books):
    books.drop_duplicates(subset='ISBN', inplace=True)
    books.dropna(subset=['ISBN', 'Book-Title', 'Book-Author'], inplace=True)
    books['Publisher'].fillna("Unknown", inplace=True)
    books['Book-Title'] = books['Book-Title'].fillna("")
    return books

def clean_users(users):
    users.drop_duplicates(subset='User-ID', inplace=True)
    users['Age'] = users['Age'].apply(lambda x: np.nan if x < 5 or x > 100 else x)
    users['Age'].fillna(users['Age'].median(), inplace=True)
    return users

def clean_ratings(ratings):
    ratings = ratings[ratings['Book-Rating'] > 0]
    ratings.drop_duplicates(inplace=True)
    return ratings
