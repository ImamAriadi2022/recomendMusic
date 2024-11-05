from flask import Flask, request, jsonify
from flask_cors import CORS
import polars as pl  # Ganti import pandas dengan polars
import joblib
import logging

# Konfigurasi logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app, resources={r"/recommend": {"origins": "*"}})

# Load data and model globally
data = None
model = None

try:
    # Load dataset dengan Polars
    data = pl.read_csv("SpotifyFeatures.csv")  
    # Load model
    model = joblib.load("model.pkl")  
    logging.info("Model and data loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or data: {str(e)}")

@app.route("/recommend", methods=["POST"])
def recommend():
    global model  # Memastikan kita menggunakan model global
    try:
        # Memeriksa apakah input ada dan valid
        if "mood" not in request.json:
            logging.error("Mood parameter is missing in request.")
            return jsonify({"error": "Mood parameter is required"}), 400
        
        user_input = request.json["mood"]  # Mendapatkan input dari frontend
        logging.debug(f"User input received: {user_input}")

        # Proses rekomendasi lagu berdasarkan `user_input`
        recommendations = model.kneighbors([user_input], n_neighbors=5)[1]

        # Mengambil data rekomendasi dari Polars
        recommended_songs = data[recommendations[0]].to_dicts()  # Menggunakan to_dicts() untuk mengonversi ke list of dicts
        
        # Mempersiapkan respons dengan CORS
        response = jsonify(recommended_songs)
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")

        return response
    except Exception as e:
        logging.error(f"Error occurred while processing request: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(debug=True)
