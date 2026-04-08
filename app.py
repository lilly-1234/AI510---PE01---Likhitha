from flask import Flask, request, jsonify
import joblib
import numpy as np

# Create Flask application
app = Flask(__name__)

# Load the trained ML model from file
model = joblib.load("model/iris_model.pkl")

# Dictionary to map numeric prediction to species name
species_map = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

# route to check if API is running
@app.route('/')
def index():
    return "MLOps Flask API is live!"

# Predict route to handle POST requests
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.get_json(force=True)

    # Extract features and convert to numpy array
    features = np.array(data['features']).reshape(1, -1)

    # Make prediction using the trained model
    prediction = model.predict(features)[0]

    # Convert numeric prediction to species name
    species = species_map[int(prediction)]

    # Return both numeric label and species name as JSON response
    return jsonify({
        'prediction': int(prediction),
        'species': species
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)