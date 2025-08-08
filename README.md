# 📚 ml-book-recommender

**Personalized, hybrid book recommendation system using collaborative filtering, NLP-based content similarity, and popularity-based models — deployed via an interactive Streamlit web app.**

---

## 📘 Project Overview

Discovering the right book can be overwhelming with millions of titles available. **`ml-book-recommender`** solves this by delivering highly personalized book suggestions based on user preferences, book content, and global popularity trends.

This end-to-end machine learning project combines:

- ✅ Collaborative Filtering (SVD)
- ✅ Natural Language Processing (TF-IDF + cosine similarity)
- ✅ Popularity-based logic

Inspired by real-world systems used by **Amazon**, **Goodreads**, and **Netflix**, this project showcases full-stack data science skills from data ingestion to deployment.

---

## 🔧 Features

- **Data Preprocessing & Cleaning** – Books, users, and ratings datasets cleaned, deduplicated, and validated.
- **Collaborative Filtering (SVD)** – Learns hidden user–book patterns for personalized recommendations.
- **Content-Based Filtering** – Suggests similar books using TF-IDF and cosine similarity on title and metadata.
- **Popularity-Based Model** – Recommends most-rated books as a cold-start fallback.
- **Hybrid Recommendation System** – Combines all three approaches for enhanced accuracy and diversity.
- **Evaluation Metrics** – RMSE for collaborative filtering performance.
- **Interactive Web UI** – Built with Streamlit for real-time recommendations and model insights.
- **Data Visualizations** – Heatmaps, PCA plots, demographic insights, and rating distributions.

---

## 📊 Model Performance

| Model                         | Metric                  | Score     |
|------------------------------|-------------------------|-----------|
| Collaborative Filtering (SVD) | RMSE                    | 8.4585    |
| Content-Based (TF-IDF)        | Similarity Match Quality| High      |
| Popularity-Based              | Coverage                | 100%      |

> ✅ The **hybrid model** outperforms individual models by combining personalization, content relevance, and popularity coverage.

---

## 🖥 Streamlit Web Application

**Main Features:**

- Select a **User ID** for personalized recommendations (SVD).
- Search by **Book Title** for content-based suggestions.
- View **evaluation metrics** like RMSE instantly.
- Fully interactive and browser-based — no web development experience needed.

---

## 📈 Example Visuals

- **Distribution of Ratings** – Understand rating tendencies.
- **Top Rated Books** – Most popular titles in the dataset.
- **User Demographics** – Age distribution & geographic insights.
- **Heatmap** – Interaction density between top users and books.
- **PCA Plot** – 2D visualization of user clusters.

---

## 🧠 Skills & Tools Demonstrated

- **Languages & Libraries:** Python, Pandas, NumPy, SciPy, Scikit-learn, Seaborn, Matplotlib  
- **Machine Learning:** SVD, content-based, popularity-based, hybrid recommendation models  
- **NLP Techniques:** TF-IDF, Cosine Similarity  
- **Evaluation Metrics:** RMSE  
- **Visualization:** Heatmaps, PCA, statistical charts  
- **Deployment:** Streamlit Web App  
- **Design Principles:** Modular, scalable, and reproducible project structure  

---

## 🎯 Why This Project Is a Great Fit for Hiring Managers

- **End-to-End Solution** – From raw data to a deployed web app  
- **Scalable Design** – Easily extensible to movies, music, or e-commerce  
- **Business Impact** – Improves discovery and user engagement through personalization  
- **Portfolio-Ready** – Demonstrates ML, NLP, EDA, model deployment, and UI skills in one showcase  

---


## 🚀 How to Run Locally

```bash
# Clone repository
git clone https://github.com/anoushirazi/ml-book-recommender.git
cd ml-book-recommender

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app/__Streamlit.py__/app.py

```

## 📂 Project Structure

```bash
├── README.md                     # Project overview and usage instructions
├── requirements.txt              # Python dependencies
│
├── data/
│   ├── raw/                      # Original datasets (excluded from Git)
│   └── processed/                # Cleaned & transformed datasets
│
├── notebooks/                    # Jupyter notebooks for each stage
│   ├── 01_Data_Preprocessing_and_Cleaning.ipynb
│   ├── 02_Exploratory_Data_Analysis.ipynb
│   ├── 03_Model_Building_SVD_TFIDF_Popularity.ipynb
│   ├── 04_Hybrid_Model_and_Evaluation.ipynb
│   └── 05_Streamlit_App_Development.ipynb
│
├── src/                          # Source code
│   ├── __init__.py
│   ├── data_utils.py
│   ├── recommender_svd.py
│   ├── recommender_content.py
│   ├── recommender_popularity.py
│   └── recommender_hybrid.py
│
├── models/                       # Saved model artifacts
│   ├── svd_model.pkl
│   ├── tfidf_matrix.pkl
│   └── popularity_scores.pkl
│
├── visuals/                      # Generated visualizations & plots
│   ├── ratings_distribution.png
│   ├── top_books.png
│   ├── user_age_distribution.png
│   ├── heatmap_top_users_books.png
│   └── pca_user_clusters.png
│
└── app/                          # Streamlit application
    ├── main.py
    └── utils.py
