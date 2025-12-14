import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "kc_house_data_cleaned.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

RANDOM_STATE = 42
TEST_SIZE = 0.2
