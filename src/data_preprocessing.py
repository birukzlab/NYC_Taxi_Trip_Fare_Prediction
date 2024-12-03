import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import requests
import pickle
import os

class DataPreprocessor:
    def __init__(self, api_url, params):
        """
        Initialize the preprocessor by fetching data from the API.
        """
        self.api_url = api_url
        self.params = params
        self.df = self.fetch_data()
        self.scaler = StandardScaler()
        
    def fetch_data(self, save_path="data/raw_data.csv"):
        """
        Fetch data from the NYC API or load it from a saved file if available.
        """
        
        if os.path.exists(save_path):
            print("Loading data from local file.")
            return pd.read_csv(save_path)
        else:
            response = requests.get(self.api_url, params=self.params)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data)
                df.to_csv(save_path, index=False)
                print(f"Data fetched and saved to {save_path}.")
                return df
            else:
                raise Exception(f"Failed to fetch data: {response.status_code}")

    def clean_data(self):
        """
        Cleans the dataset by:
        - Converting data types
        - Removing outliers based on fare amount and trip distance thresholds
        """
        # Convert datetime columns to datetime type
        self.df["tpep_pickup_datetime"] = pd.to_datetime(self.df["tpep_pickup_datetime"])
        self.df["tpep_dropoff_datetime"] = pd.to_datetime(self.df["tpep_dropoff_datetime"])

        # Convert numeric columns to appropriate types
        self.df['trip_distance'] = pd.to_numeric(self.df['trip_distance'], errors='coerce')
        self.df['fare_amount'] = pd.to_numeric(self.df['fare_amount'], errors='coerce')
        self.df['PULocationID'] = pd.to_numeric(self.df['PULocationID'], errors='coerce')
        self.df['DOLocationID'] = pd.to_numeric(self.df['DOLocationID'], errors='coerce')
        self.df['passenger_count'] = pd.to_numeric(self.df['passenger_count'], errors='coerce')

        # Add new columns: hour and day_of_week
        self.df["hour"] = self.df["tpep_pickup_datetime"].dt.hour
        self.df["day_of_week"] = self.df["tpep_pickup_datetime"].dt.dayofweek

        # Remove rows where fare amount or trip distance are unrealistic
        self.df = self.df[
            (self.df["fare_amount"] >= 2.5) & 
            (self.df["fare_amount"] <= 200) & 
            (self.df["trip_distance"] > 0) & 
            (self.df["trip_distance"] <= 100)
        ]

        print("Data cleaned: Unrealistic fare amounts and distances capped.")

    
    def add_location_frequencies(self, save_dir="models/"):
        """
        Adds `PU_freq` and `DO_freq` columns, representing the frequency
        of pickup and dropoff locations, and saves the mappings.
        """
        # Calculate frequencies for pickup and dropoff locations
        PU_freq_mapping = self.df["PULocationID"].value_counts().to_dict()
        DO_freq_mapping = self.df["DOLocationID"].value_counts().to_dict()

        # Add frequencies to the dataset
        self.df["PU_freq"] = self.df["PULocationID"].map(PU_freq_mapping)
        self.df["DO_freq"] = self.df["DOLocationID"].map(DO_freq_mapping)
        

        print("Location frequencies added and saved: 'PU_freq' and 'DO_freq'.")
    
    def add_interaction_terms(self):
        """
        Adds interaction terms to the dataset.
        """
        self.df["distance_hour"] = self.df["trip_distance"] * self.df["hour"]
        self.df["PU_DO_interaction"] = self.df["PU_freq"] * self.df["DO_freq"]
        print("Interaction terms added.")
    
    def add_clusters(self, n_clusters=10, save_dir="models/"):
        # Ensure KMeans is trained on NumPy arrays without headers
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)

        # Fit the model and assign clusters
        self.df["PU_cluster"] = kmeans.fit_predict(self.df[["PU_freq"]].values)
        self.df["DO_cluster"] = kmeans.fit_predict(self.df[["DO_freq"]].values)

        print(f"Clusters added: {n_clusters} clusters created and model saved.")

    def add_rush_hour_flag(self):
        """
        Adds a binary flag for rush hour trips.
        """
        def is_rush_hour(hour):
            return 1 if (7 <= hour <= 9) or (16 <= hour <= 18) else 0
        
        self.df["rush_hour"] = self.df["hour"].apply(is_rush_hour)
        print("Rush hour flag added.")
    
    def select_features(self):
        """
        Returns the feature matrix (X) and target variable (y).
        """
        X = self.df[[
            "trip_distance",
            "hour",
            "day_of_week",
            "rush_hour"
        ]]
        y = self.df["fare_amount"]
        print("Feature selection complete.")
        return X, y
    
    def scale_features(self, X):
        """
        Scales the features using StandardScaler.
        """
        X_scaled = self.scaler.fit_transform(X)
        print("Features scaled.")
        return X_scaled
