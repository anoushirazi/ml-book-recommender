# src/data/data_loader.py

import pandas as pd

def load_books(path="books.csv"):
    return pd.read_csv(path, low_memory=False)

def load_users(path="users.csv"):
    return pd.read_csv(path)

def load_ratings(path="ratings.csv"):
    return pd.read_csv(path)
