import os
import pickle
import pandas as pd

from config import MODEL_DIR, DATA_PATH


def load_model_and_scaler():
    model_path = os.path.join(MODEL_DIR, "gradient_boosting.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Run train.py first.")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Scaler not found. Run train.py first.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


def get_feature_columns():
    df = pd.read_csv(DATA_PATH)
    return df.drop("price", axis=1).columns.tolist()


def predict_price(input_dict):
    model, scaler = load_model_and_scaler()
    feature_columns = get_feature_columns()

    # Build DataFrame with correct feature names & order
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_columns]

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    return prediction


if __name__ == "__main__":
    # Example input (dictionary is SAFEST)
    example_house = {
        "bedrooms": 3,
        "bathrooms": 2,
        "sqft_living": 1800,
        "sqft_lot": 5000,
        "floors": 1,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "grade": 7,
        "sqft_above": 1800,
        "sqft_basement": 0,
        "yr_built": 2014,
        "yr_renovated": 0,
        "zipcode": 98052,
        "lat": 47.5,
        "long": -122.2,
        "sqft_living15": 1800,
        "sqft_lot15": 5000,
        "year": 2014,
        "month": 5
    }

    price = predict_price(example_house)
    print("🏠 Predicted House Price:", round(price, 2))
