import pandas as pd

def preprocess_data(df):
    df = df.copy()

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Convert date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df.drop(columns=["date", "id"], inplace=True)

    # Fill missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    X = df.drop("price", axis=1)
    y = df["price"]

    return X, y
