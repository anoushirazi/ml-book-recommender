# 📚 ml-book-recommender

**Personalized, hybrid book recommendation system using collaborative filtering, NLP-based content similarity, and popularity-based models — deployed via an interactive Streamlit web app.**

---

## 🎬 Demo

![Demo Preview](https://github.com/anoushirazi/ml-book-recommender/raw/main/Streamlit_Demo.mp4)

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

## 🎯 End-to-End Scalable Personalization Showcase

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
streamlit run app/app.py

```

## 📂 Main Project Structure

```bash
ml-book-recommender/                   # Root project folder 
├── README.md                          # Project overview and instructions 
├── requirements.txt                   # Python dependencies list 
├── .gitignore                         #  Git ignore rules 
├── Demo.gif                           # Demo Preview
├── LICENSE                            # Project license 

├── 📁 data/                          # Data folder 
│   ├── raw/                           # Raw/unprocessed data 
│   │   ├── books.csv                  #  Books dataset 
│   │   ├── users.csv                  #  Users dataset 
│   │   └── ratings.csv                #  Ratings dataset
│   ├── processed/                     #  Cleaned/processed data 
│   │   ├── books_cleaned.csv   
│   │   ├── users_cleaned.csv    
│   │   └── ratings_filtered.csv  
│   └── sample/                        # Sample data for quick tests
│       ├── sample_books.csv      
│       ├── sample_users.csv     
│       └── sample_ratings.csv   

├── 📓 notebooks/
    ├── 00_main.ipynb                   # Jupyter notebooks for exploration & modeling 
│   ├── 01_data_exploration.ipynb   
│   ├── 02_data_preprocessing.ipynb  
│   ├── 03_model_development.ipynb   
│   ├── 04_model_evaluation.ipynb    
│   └── 05_visualization.ipynb       

└── 🏗️ src/                            #  Source code folder
    ├── __init__.py                     #  Package init 
    ├── data/                           #  Data handling modules 
    │   ├── __init__.py        
    │   ├── data_loader.py               #  Load data scripts 
    │   ├── data_cleaner.py              # Cleaning scripts 
    │   └── data_preprocessor.py         # Preprocessing scripts
    └── models/                          #  Model implementations 
        ├── __init__.py        
        ├── collaborative_filtering.py  
        ├── content_based.py    
        ├── popularity_based.py 
        └── hybrid_model.py     

  Application & Assets 
📱 app/                                 # Streamlit app folder 
├── app.py                              # Main Streamlit app entry
├── components/                         # UI components 
│   ├── __init__.py             
│   ├── sidebar.py                      # Sidebar component 
│   ├── recommendation_display.py       #  Show recommendations 
│   └── evaluation_display.py           #  Show evaluation metrics 
└── assets/
    ├── Streamlit-web-app.png            #  Streamlit platform shot
    ├── Streamlit_Demo.rar               #  Streamlit demo archive
    └── project presentation             # Streamlit project presentation slides 

🤖 models/                             #  Saved model files 
├── svd_model.pkl              
├── tfidf_vectorizer.pkl       
└── user_item_matrix.pkl       

📊 plots/                              #  Visualization images
├── eda/                                #  Exploratory Data Analysis 
│   ├── rating_distribution.png
│   ├── user_activity.png      
│   ├── publication_trends.png 
│   └── age_demographics.png   
├── model_performance/                 # Model evaluation plots 
│   ├── rmse_comparison.png    
│   ├── precision_recall.png   
│   └── recommendation_accuracy.png
└── visualizations/                     # Other visualizations 
    ├── user_book_heatmap.png  
    ├── pca_user_clusters.png  
    └── similarity_matrix.png  

🧪 tests/                              # Unit and integration tests
├── __init__.py               
├── test_data_processing.py   
├── test_models.py            
└── test_evaluation.py        

⚙️ config/                            # Configuration files 
├── model_config.yaml          
├── data_config.yaml           
└── app_config.yaml            
