# app/components/recommendation_display.py

import streamlit as st

def show_recommendations(content_df, collab_df, pop_df):
    st.subheader("🔹 Content-Based Recommendations")
    st.dataframe(content_df[['Book-Title', 'Book-Author']].reset_index(drop=True).head(5))

    st.subheader("🔹 Collaborative Filtering Recommendations")
    st.dataframe(collab_df[['Book-Title', 'Book-Author']].reset_index(drop=True).head(5))

    st.subheader("🔹 Popular Books")
    st.dataframe(pop_df[['Book-Title', 'Book-Author']].reset_index(drop=True).head(5))
