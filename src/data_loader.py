import pandas as pd
from config import DATA_PATH

def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except FileNotFoundError:
        raise Exception("Dataset not found. Check data path.")
