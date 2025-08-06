import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from math import sqrt
import warnings

warnings.filterwarnings("ignore")

# Streamlit page settings
st.set_page_config(layout="wide", page_icon="📚", page_title="BookWise Recommender")

# Load datasets
@st.cache_data
def load_data():
    books = pd.read_csv("C:/Users/hh/Desktop/books.csv", low_memory=False)
    users = pd.read_csv("C:/Users/hh/Desktop/users.csv")
    ratings = pd.read_csv("C:/Users/hh/Desktop/ratings.csv")

    # Preprocessing
    books.drop_duplicates(subset='ISBN', inplace=True)
    books.dropna(subset=['ISBN', 'Book-Title', 'Book-Author'], inplace=True)
    books['Publisher'].fillna("Unknown", inplace=True)
    books['Book-Title'].fillna("", inplace=True)
    ratings = ratings[ratings['Book-Rating'] > 0].drop_duplicates()

    return books, users, ratings

books, users, ratings = load_data()

# Filter only books that have ratings
valid_books = ratings['ISBN'].unique()
books = books[books['ISBN'].isin(valid_books)]

# Sidebar inputs
st.sidebar.header("Settings")
user_id = st.sidebar.number_input("User ID:", min_value=1, value=1)
book_options = books['Book-Title'].dropna().unique().tolist()
selected_book = st.sidebar.selectbox("Select a Book Title:", book_options, index=0)

# App header
st.title("📚 BookWise Recommender")
st.write("Discover personalized book recommendations using content-based filtering, collaborative filtering (SVD), and popularity metrics.")

# --- Recommendation Algorithms ---

def popularity_recommender(ratings, books, top_n=10):
    popular = ratings.groupby('ISBN')['Book-Rating'].count().sort_values(ascending=False).head(top_n)
    return books[books['ISBN'].isin(popular.index)]

def content_based_recommender(book_title, books, top_n=10):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(books['Book-Title'])

    nn = NearestNeighbors(n_neighbors=top_n + 1, metric='cosine', algorithm='brute')
    nn.fit(tfidf_matrix)

    # Fix: Use positional indices for TF-IDF matrix rows
    indices = pd.Series(data=range(len(books)), index=books['Book-Title']).drop_duplicates()

    if book_title not in indices:
        st.warning("Book not found.")
        return pd.DataFrame()

    idx = indices[book_title]
    distances, neighbors = nn.kneighbors(tfidf_matrix[idx])
    similar_indices = neighbors.flatten()[1:]  # skip self
    return books.iloc[similar_indices]

def svd_recommender(ratings, user_id, books, top_n=10, k=20, sample_users=5000, sample_books=3000):
    user_counts = ratings['User-ID'].value_counts()
    eligible_users = user_counts[user_counts >= 5].index.tolist()

    if user_id not in eligible_users:
        st.warning("User ID not found or has insufficient data.")
        return pd.DataFrame()

    eligible_users_set = set(eligible_users)
    sampled_users = np.random.choice(
        list(eligible_users_set - {user_id}),
        min(sample_users - 1, len(eligible_users_set) - 1),
        replace=False
    ).tolist()
    sampled_users.append(user_id)  # Ensure user is included

    ratings_sampled = ratings[ratings['User-ID'].isin(sampled_users)]

    book_counts = ratings_sampled['ISBN'].value_counts()
    sampled_books = book_counts.index[:sample_books]

    ratings_sampled = ratings_sampled[ratings_sampled['ISBN'].isin(sampled_books)]

    user_item_matrix = ratings_sampled.pivot(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)

    user_mean = user_item_matrix.mean(axis=1)
    R_demeaned = user_item_matrix.sub(user_mean, axis=0)

    R_sparse = csr_matrix(R_demeaned.values)
    k = min(k, min(R_sparse.shape) - 1)

    U, sigma, Vt = svds(R_sparse, k=k)
    sigma = np.diag(sigma)
    all_predicted = np.dot(np.dot(U, sigma), Vt) + user_mean.values[:, np.newaxis]

    preds_df = pd.DataFrame(all_predicted, columns=user_item_matrix.columns, index=user_item_matrix.index)

    sorted_preds = preds_df.loc[user_id].sort_values(ascending=False)
    read_books = ratings_sampled[ratings_sampled['User-ID'] == user_id]['ISBN'].values
    recommendations = sorted_preds[~sorted_preds.index.isin(read_books)].head(top_n)

    return books[books['ISBN'].isin(recommendations.index)]

def hybrid_recommender(user_id, book_title, books, ratings):
    st.subheader("🔹 Content-Based Recommendations")
    try:
        content_recs = content_based_recommender(book_title, books)
        if not content_recs.empty:
            st.table(content_recs[["Book-Title", "Book-Author", "Publisher"]])
    except Exception as e:
        st.error(f"Content-based error: {e}")

    st.subheader("🔹 Collaborative Filtering (SVD) Recommendations")
    try:
        svd_recs = svd_recommender(ratings, user_id, books)
        if not svd_recs.empty:
            st.table(svd_recs[["Book-Title", "Book-Author", "Publisher"]])
    except Exception as e:
        st.error(f"SVD error: {e}")

    st.subheader("🔹 Popular Books")
    try:
        pop_recs = popularity_recommender(ratings, books)
        st.table(pop_recs[["Book-Title", "Book-Author", "Publisher"]])
    except Exception as e:
        st.error(f"Popularity error: {e}")

def evaluate_split_small(ratings, sample_size=5000, k=20):
    sample = ratings.sample(sample_size, random_state=42)
    train, test = train_test_split(sample, test_size=0.2, random_state=42)

    common_users = list(set(train['User-ID']) & set(test['User-ID']))
    if not common_users:
        st.error("No overlapping users. Try increasing sample size.")
        return None

    train = train[train['User-ID'].isin(common_users)]
    test = test[test['User-ID'].isin(common_users)]
    train_matrix = train.pivot(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)

    if train_matrix.shape[0] < 2 or train_matrix.shape[1] < 2:
        st.error("Insufficient data for evaluation.")
        return None

    user_mean = train_matrix.mean(axis=1)
    R_demeaned = train_matrix.sub(user_mean, axis=0)

    R_sparse = csr_matrix(R_demeaned.values)
    k = min(k, min(R_sparse.shape) - 1)

    U, sigma, Vt = svds(R_sparse, k=k)
    sigma = np.diag(sigma)
    preds = np.dot(np.dot(U, sigma), Vt) + user_mean.values[:, np.newaxis]
    preds_df = pd.DataFrame(preds, columns=train_matrix.columns, index=train_matrix.index)

    # Rename columns for safe attribute access
    test_renamed = test.rename(columns={'User-ID': 'User_ID', 'Book-Rating': 'Book_Rating'})

    y_true, y_pred = [], []

    for row in test_renamed.itertuples(index=False):
        user = row.User_ID
        isbn = row.ISBN
        true_rating = row.Book_Rating
        if user in preds_df.index and isbn in preds_df.columns:
            y_true.append(true_rating)
            y_pred.append(preds_df.loc[user, isbn])

    return sqrt(mean_squared_error(y_true, y_pred)) if y_true else None

# --- UI Buttons ---
if st.button("Show Recommendations"):
    if user_id not in users['User-ID'].values:
        st.error("Invalid User ID. Try one that exists in the dataset.")
    else:
        with st.spinner("Generating recommendations..."):
            hybrid_recommender(user_id, selected_book, books, ratings)

if st.button("Evaluate RMSE"):
    with st.spinner("Evaluating model..."):
        rmse = evaluate_split_small(ratings)
        if rmse is not None:
            st.success(f"RMSE: {rmse:.4f}")
        else:
            st.error("Could not compute RMSE.")

# --- Footer ---
st.markdown("---")
st.write("Developed by [Anoush A. Shirazi](https://github.com/anoushirazi) | Updated: Jul 22, 2025")
