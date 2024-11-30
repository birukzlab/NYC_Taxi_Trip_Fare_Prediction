# model_training.py

import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from data_preprocessing import DataPreprocessor
from config import API_URL, API_PARAMS, RANDOM_FOREST_PARAMS, MODEL_PATH

def train_model():
    # Step 1: Fetch and preprocess the data
    preprocessor = DataPreprocessor(api_url=API_URL, params=API_PARAMS)
    preprocessor.clean_data()
    preprocessor.add_location_frequencies()
    preprocessor.add_interaction_terms()
    preprocessor.add_clusters(n_clusters=10)
    preprocessor.add_rush_hour_flag()
    X, y = preprocessor.select_features()
    X_scaled = preprocessor.scale_features(X)

    # Step 2: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Step 3: Train the Random Forest model
    rf_model = RandomForestRegressor(**RANDOM_FOREST_PARAMS)
    rf_model.fit(X_train, y_train)

    # Step 4: Save the trained model
    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(rf_model, model_file)
        print(f"Model saved at {MODEL_PATH}")

    return X_test, y_test  # Return test data for evaluation purposes

if __name__ == "__main__":
    train_model()
