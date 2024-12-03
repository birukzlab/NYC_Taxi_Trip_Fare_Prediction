# config.py

# NYC Taxi Dataset API
API_URL = "https://data.cityofnewyork.us/resource/m6nq-qud6.json"
API_PARAMS = {
    "$limit": 100000,
    "$select": "tpep_pickup_datetime, tpep_dropoff_datetime, trip_distance, fare_amount, PULocationID, DOLocationID, passenger_count"
}

# Model Training Parameters - Used Gridsearch to find the best Parameters
RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 10,
    "min_samples_split": 10,
    "min_samples_leaf": 1,
    "random_state": 42
}

# Paths for Models and Data
MODEL_PATH = "models/random_forest.pkl"
