from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = joblib.load("fraud_detection.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read all input fields V1–V10
        features = [float(request.form[f"V{i}"]) for i in range(1, 11)]
        
        # Amount and Time fields
        amount = float(request.form["amount"])
        time = float(request.form["time"])

        # Normalize amount and time
        scaler = StandardScaler()
        normalized = scaler.fit_transform([[amount, time]])
        norm_amount, norm_time = normalized[0]

        # Final input for prediction
        final_input = np.array(features + [norm_amount, norm_time]).reshape(1, -1)

        prediction = model.predict(final_input)[0]
        result = "⚠️ Fraudulent Transaction Detected" if prediction == 1 else "✅ Genuine Transaction"
    
    except Exception as e:
        print("Error:", e)
        result = "❌ Error: Please fill all values correctly."

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
