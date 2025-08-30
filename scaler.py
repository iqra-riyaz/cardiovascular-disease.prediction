import joblib

def load_scaler():
    return joblib.load("scaler.pkl")