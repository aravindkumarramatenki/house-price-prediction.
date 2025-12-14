import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import time

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# -------------------------------------------------
# Dark mode toggle
# -------------------------------------------------
dark_mode = st.toggle("🌙 Dark Mode")

if dark_mode:
    bg_color = "#0e1117"
    card_color = "#1c1f26"
    text_color = "#ffffff"
else:
    bg_color = "#f5f7fa"
    card_color = "#ffffff"
    text_color = "#000000"

# -------------------------------------------------
# Custom CSS
# -------------------------------------------------
st.markdown(f"""
<style>
html {{ scroll-behavior: smooth; }}
body {{ background-color: {bg_color}; color: {text_color}; }}

.card {{
    background: {card_color};
    padding: 1.5rem;
    border-radius: 16px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.12);
    margin-bottom: 1.5rem;
}}

.result-box {{
    background: linear-gradient(120deg, #4CAF50, #2ECC71);
    color: white;
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
    margin-top: 2rem;
}}

.feature-box {{
    background: {card_color};
    padding: 1rem;
    border-radius: 14px;
    margin-top: 1rem;
}}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "kc_house_data_cleaned.csv")

# -------------------------------------------------
# Load model & scaler
# -------------------------------------------------
@st.cache_resource
def load_model():
    with open(os.path.join(MODEL_DIR, "gradient_boosting.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_data
def load_features():
    df = pd.read_csv(DATA_PATH)
    return df.drop("price", axis=1).columns.tolist()

model, scaler = load_model()
features = load_features()

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>🏠 House Price Predictor</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Premium ML-based house price estimation</p>",
    unsafe_allow_html=True
)

# -------------------------------------------------
# Sample data button
# -------------------------------------------------
if "sample" not in st.session_state:
    st.session_state.sample = False

if st.button("⚡ Use Sample Data"):
    st.session_state.sample = True

# -------------------------------------------------
# Input Section
# -------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🏗️ Property Details")

col1, col2, col3 = st.columns(3)
user_input = {}

def sample(val, default):
    return default if st.session_state.sample else val

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

# -------------------------------------------------
# Location Section
# -------------------------------------------------
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

# -------------------------------------------------
# Neighborhood Section
# -------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🏘️ Neighborhood Averages")

user_input["sqft_living15"] = st.number_input("Avg Living Area (15 nearby)", 300, 10000, sample(2000, 2200))
user_input["sqft_lot15"] = st.number_input("Avg Lot Size (15 nearby)", 300, 50000, sample(6000, 6500))

user_input["year"] = user_input["yr_built"]
user_input["month"] = 6

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("🔮 Predict Price"):
    with st.spinner("Estimating house price..."):
        time.sleep(0.6)

        input_df = pd.DataFrame([user_input])
        input_df = input_df[features]

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

    # -------------------------------------------------
    # Feature insight (trust builder)
    # -------------------------------------------------
    st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
    st.subheader("📊 What influenced this price most?")
    st.markdown("""
    • Living area (sqft_living)  
    • Location (latitude & longitude)  
    • Construction quality (grade)  
    • Waterfront & view  
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Built with ❤️ using Streamlit & Machine Learning")
