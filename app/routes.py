from flask import Blueprint, render_template, request
import pickle
import numpy as np


main = Blueprint("main", __name__)

# Load the trained Random Forest model
model_path = "models/random_forest.pkl"
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

@main.route('/')
def index():
    """
    Render the homepage.
    """
    return render_template("index.html")

@main.route('/predict', methods=['POST'])
def predict():
    """
    Handle fare prediction based on user inputs.
    """
    try:
        # Collect user input from the form
        trip_distance = float(request.form.get("trip_distance"))
        hour = int(request.form.get("hour"))
        day_of_week = int(request.form.get("day_of_week"))
        rush_hour = 1 if int(request.form.get("rush_hour")) == 1 else 0

        # Create a feature vector for the model
        feature_vector = np.array([trip_distance, hour, day_of_week, rush_hour]).reshape(1, -1)

        # Make prediction
        predicted_fare = model.predict(feature_vector)[0]

        return render_template("result.html", fare_amount=round(predicted_fare, 2))

    except Exception as e:
        print(f"Error occurred: {e}")
        return render_template("error.html", error_message="An error occurred while processing your request. Please try again.")







