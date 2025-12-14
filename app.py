import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import time

# --------------------------------------------------
# BASIC PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction")
st.write("Machine Learning based house price estimator")

# --------------------------------------------------
# FIXED, SIMPLE PATHS (STREAMLIT CLOUD STANDARD)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "gradient_boosting.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "kc_house_data_cleaned.csv")

# --------------------------------------------------
# HARD CHECKS (CLEAR ERRORS)
# --------------------------------------------------
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found: models/gradient_boosting.pkl")
    st.stop()

if not os.path.exists(SCALER_PATH):
    st.error("❌ Scaler file not found: models/scaler.pkl")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error("❌ Data file not found: data/kc_house_data_cleaned.csv")
    st.stop()

# --------------------------------------------------
# LOAD MODEL, SCALER, FEATURES (NO CACHE)
# --------------------------------------------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

df = pd.read_csv(DATA_PATH)
feature_columns = df.drop("price", axis=1).columns.tolist()

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
st.subheader("Enter house details")

col1, col2, col3 = st.columns(3)
user_input = {}

with col1:
    user_input["bedrooms"] = st.number_input("Bedrooms", 1, 10, 3)
    user_input["bathrooms"] = st.number_input("Bathrooms", 1, 8, 2)
    user_input["floors"] = st.number_input("Floors", 1, 3, 1)

with col2:
    user_input["sqft_living"] = st.number_input("Sqft Living", 300, 10000, 1800)
    user_input["sqft_lot"] = st.number_input("Sqft Lot", 500, 50000, 5000)
    user_input["sqft_above"] = st.number_input("Sqft Above", 300, 10000, 1800)

with col3:
    user_input["sqft_basement"] = st.number_input("Sqft Basement", 0, 5000, 0)
    user_input["condition"] = st.number_input("Condition (1–5)", 1, 5, 3)
    user_input["grade"] = st.number_input("Grade (1–13)", 1, 13, 7)

st.subheader("Location")

col4, col5, col6 = st.columns(3)

with col4:
    user_input["zipcode"] = st.number_input("Zipcode", value=98052)
    user_input["lat"] = st.number_input("Latitude", value=47.68)
    user_input["long"] = st.number_input("Longitude", value=-122.12)

with col5:
    user_input["waterfront"] = st.selectbox("Waterfront", [0, 1])
    user_input["view"] = st.selectbox("View", [0, 1, 2, 3, 4])

with col6:
    user_input["yr_built"] = st.number_input("Year Built", 1900, 2025, 2015)
    user_input["yr_renovated"] = st.number_input("Year Renovated", 0, 2025, 0)

st.subheader("Neighborhood")

user_input["sqft_living15"] = st.number_input("Sqft Living 15", 300, 10000, 2000)
user_input["sqft_lot15"] = st.number_input("Sqft Lot 15", 300, 50000, 6000)
user_input["year"] = user_input["yr_built"]
user_input["month"] = 6

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if st.button("Predict Price"):
    with st.spinner("Predicting price..."):
        time.sleep(0.5)

        input_df = pd.DataFrame([user_input])
        input_df = input_df[feature_columns]

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        low = prediction * 0.92
        high = prediction * 1.08

    st.success(f"Estimated Price: ₹ {prediction:,.2f}")
    st.info(f"Likely Range: ₹ {low:,.2f} – ₹ {high:,.2f}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("Deployed on Streamlit Cloud • No cache • Stable build")
