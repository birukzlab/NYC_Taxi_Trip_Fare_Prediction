from flask import Blueprint, render_template, request, jsonify
import pickle
import numpy as np

main = Blueprint('main', __name__)

# Load the trained Random Forest model
model = pickle.load(open('models/random_forest.pkl', 'rb'))

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    # Extract features from form submission
    data = request.form
    features = np.array([
        float(data['trip_distance']),
        int(data['hour']),
        int(data['day_of_week']),
        float(data['PU_freq']),
        float(data['DO_freq']),
        float(data['distance_hour']),
        float(data['PU_DO_interaction']),
        int(data['PU_cluster']),
        int(data['DO_cluster']),
        int(data['rush_hour'])
    ]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)
    return render_template('result.html', prediction=round(prediction[0], 2))
