from flask import Flask, request, jsonify
from flask_cors import CORS
import polars as pl
import joblib
import requests
import logging
import os

# Konfigurasi logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)

# Link Google Drive untuk model dan dataset
MODEL_URL = "https://drive.google.com/uc?export=download&id=1WKIbyp-dqRVNE7602teHVHnKzQ0Elo9m"
DATASET_URL = "https://drive.google.com/uc?export=download&id=1-8CQ7lqRJk-VMkFy4h8nUE0X-C2VFazq"

# Nama file lokal untuk model dan dataset
MODEL_FILE = "model.pkl"
DATASET_FILE = "SpotifyFeatures.csv"

# Fungsi untuk mengunduh file dari Google Drive jika belum ada
def download_file(url, filename):
    if not os.path.exists(filename):
        logging.info(f"Downloading {filename}...")
        response = requests.get(url)
        with open(filename, "wb") as f:
            f.write(response.content)
        logging.info(f"{filename} downloaded successfully.")
    else:
        logging.info(f"{filename} already exists. Skipping download.")

# Download model dan dataset
try:
    download_file(MODEL_URL, MODEL_FILE)
    download_file(DATASET_URL, DATASET_FILE)

    # Load dataset dan model
    logging.info("Loading dataset...")
    data = pl.read_csv(DATASET_FILE)
    logging.info("Dataset loaded successfully.")
    
    logging.info("Loading model...")
    model = joblib.load(MODEL_FILE)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or data: {str(e)}")
    data = None
    model = None

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
