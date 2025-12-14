import pandas as pd
import os

RAW_PATH = "data/kc_house_data.csv"
CLEAN_PATH = "data/kc_house_data_cleaned.csv"

# 1. Load raw data (robust delimiter handling)
df = pd.read_csv(
    RAW_PATH,
    sep=None,
    engine="python"
)

# 2. Drop duplicates
df = df.drop_duplicates().reset_index(drop=True)

# 3. Date processing
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

# 4. Drop unused columns
df = df.drop(columns=["id", "date"])

# 5. Ensure correct dtypes
numeric_cols = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "grade",
    "sqft_above", "sqft_basement",
    "yr_built", "yr_renovated",
    "zipcode", "lat", "long",
    "sqft_living15", "sqft_lot15",
    "year", "month"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 6. Handle missing values
df = df.fillna(df.median(numeric_only=True))

# 7. Remove extreme price outliers (1%–99%)
q_low = df["price"].quantile(0.01)
q_high = df["price"].quantile(0.99)
df = df[(df["price"] >= q_low) & (df["price"] <= q_high)]

# 8. Save cleaned dataset
os.makedirs("data", exist_ok=True)
df.to_csv(CLEAN_PATH, index=False)

print("✅ kc_house_data_cleaned.csv created successfully")
print("Final shape:", df.shape)
print("Columns:", list(df.columns))
