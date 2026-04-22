from flask import Flask, request, jsonify
from flask_cors import CORS

# 🔥 QUICK FIX for pickle error
import numpy as np
class ManualScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def transform(self, X):
        return (X - self.mean) / self.std

# Import your backend functions
from Backend import predict_fos, classify_risk, sensitivity_analysis

app = Flask(__name__)
CORS(app)

# ==============================
# HOME ROUTE
# ==============================
@app.route("/")
def home():
    return "Slope Stability API Running 🚀"

# ==============================
# PREDICTION ROUTE
# ==============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        fos = predict_fos(data)
        risk = classify_risk(fos)

        return jsonify({
            "fos": fos,
            "risk": risk
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400

# ==============================
# SENSITIVITY ROUTE
# ==============================
@app.route("/sensitivity", methods=["POST"])
def sensitivity():
    try:
        data = request.json
        variable = data.get("variable")
        if not variable:
            return jsonify({"error": "Variable is required"}), 400

        x, y = sensitivity_analysis(data, variable)

        return jsonify({
            "x": x,
            "y": y
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400

# ==============================
# RUN SERVER
# ==============================
if __name__ == "__main__":
    app.run()
    
    
    
    
    