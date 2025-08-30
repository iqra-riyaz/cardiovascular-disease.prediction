from flask import Flask, request, jsonify,render_template
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and scaler
model = load_model("cardio_model.h5")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Collect features in exact order
    features = [
        int(data["age"]),
        1 if data["gender"] == "male" else 0,
        int(data["ap_hi"]),
        int(data["ap_lo"]),
        0 if int(data["cholesterol"]) == 1 else 1,  # normalize like training
        0 if int(data["gluc"]) == 1 else 1,
        float(data["bmi"]),
        1 if data["smoke"] == "yes" else 0,
        1 if data["alco"] == "yes" else 0,
        1 if data["active"] == "yes" else 0
    ]

    # Scale input
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0][0]
    prediction = 1 if prediction >= 0.5 else 0

    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
