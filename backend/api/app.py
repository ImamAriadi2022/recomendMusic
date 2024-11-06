from flask import Flask, request, jsonify
from flask_cors import CORS
import polars as pl
import joblib
import logging

# Konfigurasi logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)

# Load data and model
data = None
model = None

try:
    logging.info("Loading data...")
    data = pl.read_csv("../SpotifyFeatures.csv")
    logging.info("Data loaded successfully.")
    
    logging.info("Loading model...")
    model = joblib.load("../model.pkl")
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or data: {str(e)}")

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        if "mood" not in request.json:
            return jsonify({"error": "Mood parameter is required"}), 400
        
        if model is None:
            return jsonify({"error": "Model not available"}), 500
        
        user_input = request.json["mood"]
        logging.debug(f"User input received: {user_input}")

        recommendations = model.kneighbors([user_input], n_neighbors=5)[1]
        recommended_songs = data[recommendations[0]].to_dicts()
        
        response = jsonify(recommended_songs)
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")

        return response
    except Exception as e:
        logging.error(f"Error occurred while processing request: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run()  # Consider removing this line for deployment
