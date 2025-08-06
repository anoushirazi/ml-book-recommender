# src/models/hybrid_model.py

from .content_based import content_based_recommender
from .collaborative_filtering import svd_recommender
from .popularity_based import popularity_recommender

def hybrid_recommender(user_id, book_title, ratings, books):
    content = content_based_recommender(book_title, books)
    collab = svd_recommender(ratings, user_id, books)
    popular = popularity_recommender(ratings, books)

    print("🔹 Content-Based Recommendations:")
    print(content[['Book-Title']].head(5))
    print("\n🔹 Collaborative Filtering Recommendations:")
    print(collab[['Book-Title']].head(5))
    print("\n🔹 Popular Books:")
    print(popular[['Book-Title']].head(5))
