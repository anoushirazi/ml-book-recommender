# app/components/evaluation_display.py

import streamlit as st

def show_evaluation_metrics(rmse_value):
    st.subheader("📊 Evaluation Metric")
    st.success(f"✅ RMSE on sample test set: {rmse_value:.4f}")
