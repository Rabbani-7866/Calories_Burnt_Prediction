import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        gender = int(request.form["gender"])
        age = float(request.form["age"])
        height = float(request.form["height"])
        weight = float(request.form["weight"])
        duration = float(request.form["duration"])
        heart_rate = float(request.form["heart_rate"])
        body_temp = float(request.form["body_temp"])

        # Create feature array
        features = np.array([[gender, age, height, weight, duration, heart_rate, body_temp]])

        # Apply the same StandardScaler before prediction
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]

        return render_template("result.html", prediction=round(prediction, 2))
    
    except Exception as e:
        return render_template("result.html", error=str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000 ,debug=True)
