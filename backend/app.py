from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import logging

# Konfigurasi logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app, resources={r"/recommend": {"origins": "*"}})

# Load data and model
try:
    data = pd.read_csv("SpotifyFeatures.csv")  # Dataset harus sesuai
    model = joblib.load("model.pkl")  # Pastikan model sudah ada
    logging.info("Model and data loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or data: {str(e)}")

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        # Memeriksa apakah input ada dan valid
        if "mood" not in request.json:
            logging.error("Mood parameter is missing in request.")
            return jsonify({"error": "Mood parameter is required"}), 400
        
        user_input = request.json["mood"]  # Mendapatkan input dari frontend
        logging.debug(f"User input received: {user_input}")

        # Proses rekomendasi lagu berdasarkan `user_input`
        recommendations = model.kneighbors([user_input], n_neighbors=5)[1]
        recommended_songs = data.iloc[recommendations[0]].to_dict(orient="records")
        
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
