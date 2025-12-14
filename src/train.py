import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import DATA_PATH, MODEL_DIR, RANDOM_STATE, TEST_SIZE


def train():
    print("🔹 Loading cleaned dataset...")
    df = pd.read_csv(DATA_PATH)

    # -----------------------------
    # Feature / Target split
    # -----------------------------
    X = df.drop("price", axis=1)
    y = df["price"]

    print(f"🔹 Dataset shape: {df.shape}")
    print(f"🔹 Features shape: {X.shape}")

    # -----------------------------
    # Train / Test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # -----------------------------
    # Scaling (NO DATA LEAKAGE)
    # -----------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -----------------------------
    # Models
    # -----------------------------
    lr_model = LinearRegression()

    gbr_model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=RANDOM_STATE
    )

    print("🔹 Training Linear Regression...")
    lr_model.fit(X_train_scaled, y_train)

    print("🔹 Training Gradient Boosting Regressor...")
    gbr_model.fit(X_train_scaled, y_train)

    # -----------------------------
    # Evaluation (VERSION-SAFE)
    # -----------------------------
    preds = gbr_model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))  # SAFE FOR ALL sklearn
    r2 = r2_score(y_test, preds)

    print("\n📊 Model Performance (Gradient Boosting)")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    # -----------------------------
    # Save Models
    # -----------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(os.path.join(MODEL_DIR, "linear_regression.pkl"), "wb") as f:
        pickle.dump(lr_model, f)

    with open(os.path.join(MODEL_DIR, "gradient_boosting.pkl"), "wb") as f:
        pickle.dump(gbr_model, f)

    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print("\n✅ Models saved successfully in /models")
    print("✅ Training pipeline completed without errors")


if __name__ == "__main__":
    train()
