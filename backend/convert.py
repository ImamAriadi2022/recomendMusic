import joblib
import json
import numpy as np

# Load model
model = joblib.load('model.pkl')

# Extract the training data (features) and other useful information
model_data = {
    'n_neighbors': model.n_neighbors,  # Number of neighbors
    'trainingSet': model._fit_X.tolist()  # List of training data (features)
}

# Save model to JSON
with open('model.json', 'w') as json_file:
    json.dump(model_data, json_file)

print("Model successfully converted to JSON.")
