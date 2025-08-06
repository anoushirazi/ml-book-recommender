# src/models/__init__.py

from .collaborative_filtering import svd_recommender
from .content_based import content_based_recommender
from .popularity_based import popularity_recommender
from .hybrid_model import hybrid_recommender

__all__ = [
    "svd_recommender",
    "content_based_recommender",
    "popularity_recommender",
    "hybrid_recommender",
]
