import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_preprocessing import DataPreprocessor
from config import MODEL_PATH, API_URL, API_PARAMS

def evaluate_model():
    # Step 1: Fetch and preprocess the data
    preprocessor = DataPreprocessor(api_url=API_URL, params=API_PARAMS)
    preprocessor.clean_data()
    preprocessor.add_rush_hour_flag()
    X, y = preprocessor.select_features()
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    # Step 2: Load train-test indices
    test_indices = np.load("data/test_indices.npy")

    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]

    # Step 3: Load the trained model
    with open(MODEL_PATH, "rb") as model_file:
        rf_model = pickle.load(model_file)
        print(f"Model loaded from {MODEL_PATH}")

    # Step 4: Make predictions
    y_pred = rf_model.predict(X_test)

    # Step 5: Calculate evaluation metrics
    metrics = {
        "RMSE": mean_squared_error(y_test, y_pred, squared=False),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }

    print("Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    evaluate_model()


