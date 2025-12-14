import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import time

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ==================================================
# SAFE BASE DIRECTORY (WORKS LOCALLY + CLOUD)
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_PATH = os.path.join(MODELS_DIR, "gradient_boosting.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
DATA_PATH = os.path.join(DATA_DIR, "kc_house_data_cleaned.csv")

# ==================================================
# HARD FAIL IF FILES ARE MISSING (CLEAR MESSAGE)
# ==================================================
if not os.path.exists(MODELS_DIR):
    st.error("❌ 'models/' folder not found in the repository.")
    st.stop()

if not os.path.exists(DATA_DIR):
    st.error("❌ 'data/' folder not found in the repository.")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file 'gradient_boosting.pkl' not found.")
    st.stop()

if not os.path.exists(SCALER_PATH):
    st.error("❌ Scaler file 'scaler.pkl' not found.")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error("❌ Cleaned dataset not found.")
    st.stop()

# ==================================================
# LOAD MODEL & DATA (NO CACHE → NO CLOUD ISSUES)
# ==================================================
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

df = pd.read_csv(DATA_PATH)
feature_columns = df.drop("price", axis=1).columns.tolist()

# ==================================================
# UI STYLES
# ==================================================
st.markdown("""
<style>
html { scroll-behavior: smooth; }

.card {
    background: #ffffff;
    padding: 1.5rem;
    border-radius: 16px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    margin-bottom: 1.5rem;
}

.result-box {
    background: linear-gradient(120deg, #4CAF50, #2ECC71);
    color: white;
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================
st.markdown("<h1 style='text-align:center;'>🏠 House Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Public ML app deployed on Streamlit Cloud</p>", unsafe_allow_html=True)

# ==================================================
# SAMPLE DATA BUTTON
# ==================================================
if "sample" not in st.session_state:
    st.session_state.sample = False

if st.button("⚡ Use Sample Data"):
    st.session_state.sample = True

def sample(user_val, default_val):
    return default_val if st.session_state.sample else user_val

# ==================================================
# INPUT SECTIONS
# ==================================================
user_input = {}

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🏗️ Property Details")

col1, col2, col3 = st.columns(3)

with col1:
    user_input["bedrooms"] = st.slider("Bedrooms", 1, 10, sample(3, 4))
    user_input["bathrooms"] = st.slider("Bathrooms", 1, 8, sample(2, 3))
    user_input["floors"] = st.selectbox("Floors", [1, 2, 3], index=sample(0, 1))

with col2:
    user_input["sqft_living"] = st.number_input("Living Area (sqft)", 300, 10000, sample(1800, 2400))
    user_input["sqft_lot"] = st.number_input("Lot Size (sqft)", 500, 50000, sample(5000, 6000))
    user_input["sqft_above"] = st.number_input("Sqft Above Ground", 300, 10000, sample(1800, 2400))

with col3:
    user_input["sqft_basement"] = st.number_input("Basement Sqft", 0, 5000, sample(0, 0))
    user_input["condition"] = st.slider("Condition (1–5)", 1, 5, sample(3, 4))
    user_input["grade"] = st.slider("Grade (1–13)", 1, 13, sample(7, 8))

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📍 Location & Features")

col4, col5, col6 = st.columns(3)

with col4:
    user_input["zipcode"] = st.number_input("Zipcode", value=sample(98052, 98052))
    user_input["lat"] = st.number_input("Latitude", value=sample(47.68, 47.67), format="%.5f")
    user_input["long"] = st.number_input("Longitude", value=sample(-122.12, -122.15), format="%.5f")

with col5:
    user_input["waterfront"] = st.selectbox("Waterfront", [0, 1], index=sample(0, 0))
    user_input["view"] = st.selectbox("View", [0, 1, 2, 3, 4], index=sample(0, 1))

with col6:
    user_input["yr_built"] = st.number_input("Year Built", 1900, 2025, sample(2015, 2018))
    user_input["yr_renovated"] = st.number_input("Year Renovated", 0, 2025, sample(0, 0))

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🏘️ Neighborhood Averages")

user_input["sqft_living15"] = st.number_input("Avg Living Area (15 nearby)", 300, 10000, sample(2000, 2200))
user_input["sqft_lot15"] = st.number_input("Avg Lot Size (15 nearby)", 300, 50000, sample(6000, 6500))
user_input["year"] = user_input["yr_built"]
user_input["month"] = 6

st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# SMART WARNINGS
# ==================================================
warnings = []

if user_input["sqft_living"] < 300:
    warnings.append("Living area seems unusually small.")

if user_input["bathrooms"] > user_input["bedrooms"] + 2:
    warnings.append("Bathrooms unusually high compared to bedrooms.")

if not (47.1 <= user_input["lat"] <= 47.9):
    warnings.append("Latitude outside typical King County range.")

if warnings:
    st.warning("⚠️ Input warnings:\n- " + "\n- ".join(warnings))

# ==================================================
# PREDICTION
# ==================================================
if st.button("🔮 Predict Price"):
    with st.spinner("Estimating house price..."):
        time.sleep(0.5)

        input_df = pd.DataFrame([user_input])
        input_df = input_df[feature_columns]

        scaled = scaler.transform(input_df)
        prediction = model.predict(scaled)[0]

        low = prediction * 0.92
        high = prediction * 1.08

    st.markdown(f"""
    <div class="result-box">
        💰 Estimated Price<br>
        ₹ {prediction:,.2f}<br><br>
        <span style="font-size:18px;">
        Likely Range: ₹ {low:,.2f} – ₹ {high:,.2f}
        </span>
    </div>
    """, unsafe_allow_html=True)

# ==================================================
# FOOTER
# ==================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Built with ❤️ | Streamlit Cloud Deployment")
#end------