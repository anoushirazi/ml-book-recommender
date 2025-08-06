# app/components/sidebar.py

import streamlit as st

def render_sidebar(user_ids, book_titles):
    st.sidebar.header("📚 BookWise Recommender")

    selected_user = st.sidebar.selectbox("Select User ID", user_ids)
    selected_book = st.sidebar.selectbox("Select Book Title", book_titles)
    show_eval = st.sidebar.checkbox("Evaluate RMSE")

    return selected_user, selected_book, show_eval
