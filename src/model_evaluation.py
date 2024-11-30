# model_evaluation.py

import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from config import MODEL_PATH
from data_preprocessing import DataPreprocessor
from config import API_URL, API_PARAMS

def evaluate_model(X_test, y_test):
    # Step 1: Load the trained model
    with open(MODEL_PATH, "rb") as model_file:
        rf_model = pickle.load(model_file)
        print(f"Model loaded from {MODEL_PATH}")

    # Step 2: Make predictions
    y_pred = rf_model.predict(X_test)

    # Step 3: Calculate evaluation metrics
    metrics = {
        "RMSE": mean_squared_error(y_test, y_pred, squared=False),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }
    print("Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    # Re-run preprocessing to get test data for evaluation
    preprocessor = DataPreprocessor(api_url=API_URL, params=API_PARAMS)
    preprocessor.clean_data()
    preprocessor.add_location_frequencies()
    preprocessor.add_interaction_terms()
    preprocessor.add_clusters(n_clusters=10)
    preprocessor.add_rush_hour_flag()
    X, y = preprocessor.select_features()
    X_scaled = preprocessor.scale_features(X)

    # Split data to get the test set
    _, X_test, _, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Evaluate the model
    evaluate_model(X_test, y_test)
