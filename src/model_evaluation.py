# model_training.py
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.data_preprocessing import DataPreprocessor
from config import API_URL, API_PARAMS, RANDOM_FOREST_PARAMS, MODEL_PATH

# Fetch and preprocess data
preprocessor = DataPreprocessor(api_url=API_URL, params=API_PARAMS)
preprocessor.clean_data()
preprocessor.add_location_frequencies()
preprocessor.add_interaction_terms()
preprocessor.add_clusters(n_clusters=10)
preprocessor.add_rush_hour_flag()
X, y = preprocessor.select_features()
X_scaled = preprocessor.scale_features(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_model = RandomForestRegressor(**RANDOM_FOREST_PARAMS)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
metrics = {
    "RMSE": mean_squared_error(y_test, y_pred, squared=False),
    "MAE": mean_absolute_error(y_test, y_pred),
    "R2": r2_score(y_test, y_pred)
}
print("Model Performance:", metrics)

# Save the trained model
with open(MODEL_PATH, "wb") as model_file:
    pickle.dump(rf_model, model_file)
    print(f"Model saved at {MODEL_PATH}")
