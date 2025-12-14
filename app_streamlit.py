import os
import pickle
import streamlit as st
import pandas as pd

st.set_page_config(page_title="House Price Prediction", page_icon="🏠")

st.write("🚀 NEW FILE: app_streamlit.py is running")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "gradient_boosting.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "kc_house_data_cleaned.csv")

st.write("Base dir:", BASE_DIR)
st.write("Files here:", os.listdir(BASE_DIR))

if not os.path.exists("models"):
    st.error("❌ models folder not found")
    st.stop()

st.write("Models folder contains:", os.listdir("models"))

if not os.path.exists(MODEL_PATH):
    st.error("❌ gradient_boosting.pkl not found")
    st.stop()

if not os.path.exists(SCALER_PATH):
    st.error("❌ scaler.pkl not found")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

df = pd.read_csv(DATA_PATH)
features = df.drop("price", axis=1).columns.tolist()

st.success("✅ Model, scaler, and data loaded successfully")

st.write("This confirms Streamlit is reading the correct files.")
