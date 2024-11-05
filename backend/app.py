from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS library
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load data and model
data = pd.read_csv("SpotifyFeatures.csv")  # Assumes dataset is preprocessed
model = joblib.load("model.pkl")  # Load pretrained model

@app.route("/recommend", methods=["POST"])
def recommend():
    user_input = request.json["mood"]  # Mood input from frontend
    
    # Filter or calculate based on `user_input`
    # Example: Select songs with `valence` value corresponding to mood

    recommendations = model.kneighbors([[user_input]], n_neighbors=5)[1]
    recommended_songs = data.iloc[recommendations[0]].to_dict(orient="records")
    return jsonify(recommended_songs)

if __name__ == "__main__":
    app.run(debug=True)
