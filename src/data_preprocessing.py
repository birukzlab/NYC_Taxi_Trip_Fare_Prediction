import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import requests

class DataPreprocessor:
    def __init__(self, api_url, params):
        """
        Initialize the preprocessor by fetching data from the API.
        """
        self.api_url = api_url
        self.params = params
        self.df = self.fetch_data()
        self.scaler = StandardScaler()
        
    def fetch_data(self):
        """
        Fetch data from the NYC API.
        """
        response = requests.get(self.api_url, params=self.params)
        if response.status_code == 200:
            data = response.json()
            print("Data fetched successfully.")
            return pd.DataFrame(data)
        else:
            raise Exception(f"Failed to fetch data: {response.status_code}")

    
    def clean_data(self):
        """
        Cleans the dataset by:
        - Converting data types
        - Removing outliers
        - Dropping missing or invalid rows
        """
        # Convert datetime columns to datetime type
        self.df["tpep_pickup_datetime"] = pd.to_datetime(self.df["tpep_pickup_datetime"])
        self.df["tpep_dropoff_datetime"] = pd.to_datetime(self.df["tpep_dropoff_datetime"])
        self.df['trip_distance'] = pd.to_numeric(self.df['trip_distance'])
        self.df['fare_amount'] = pd.to_numeric(self.df['fare_amount'])
        self.df['PULocationID'] = pd.to_numeric(self.df['PULocationID'])
        self.df['DOLocationID'] = pd.to_numeric(self.df['DOLocationID'])
        self.df['passenger_count'] = pd.to_numeric(self.df['passenger_count'])

        # hour and day_of_week
        self.df.loc[:, "hour"] = self.df["tpep_pickup_datetime"].dt.hour
        self.df.loc[:, 'day_of_week'] = self.df['tpep_pickup_datetime'].dt.dayofweek
        
        # Remove rows where trip distance or fare amount are unrealistic
        self.df = self.df[(self.df["trip_distance"] > 0) & (self.df["fare_amount"] > 0)]
        
        # Remove outliers using Interquartile Range (IQR)
        for column in ["trip_distance", "fare_amount"]:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        
        print("Data cleaned: Outliers removed, and data types updated.")
    
    def add_location_frequencies(self):
        """
        Adds `PU_freq` and `DO_freq` columns, representing the frequency
        of pickup and dropoff locations.
        """
        # Calculate frequencies for pickup and dropoff locations
        self.df["PU_freq"] = self.df["PULocationID"].map(self.df["PULocationID"].value_counts())
        self.df["DO_freq"] = self.df["DOLocationID"].map(self.df["DOLocationID"].value_counts())
        print("Location frequencies added: 'PU_freq' and 'DO_freq'.")
    
    def add_interaction_terms(self):
        """
        Adds interaction terms to the dataset.
        """
        self.df["distance_hour"] = self.df["trip_distance"] * self.df["hour"]
        self.df["PU_DO_interaction"] = self.df["PU_freq"] * self.df["DO_freq"]
        print("Interaction terms added.")
    
    def add_clusters(self, n_clusters=10):
        """
        Adds pickup and dropoff clusters using KMeans.
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.df["PU_cluster"] = kmeans.fit_predict(self.df[["PU_freq"]])
        self.df["DO_cluster"] = kmeans.fit_predict(self.df[["DO_freq"]])
        print(f"Clusters added: {n_clusters} clusters created.")
    
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
            "PU_freq",
            "DO_freq",
            "distance_hour",
            "PU_DO_interaction",
            "PU_cluster",
            "DO_cluster",
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
