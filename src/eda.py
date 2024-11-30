# eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import DataPreprocessor
from config import API_URL, API_PARAMS

# Fetch and clean data for EDA
preprocessor = DataPreprocessor(api_url=API_URL, params=API_PARAMS)
preprocessor.clean_data()
preprocessor.add_location_frequencies()
preprocessor.add_interaction_terms()
preprocessor.add_clusters()
preprocessor.add_rush_hour_flag()
df = preprocessor.df  # Access the cleaned dataframe

# Define EDA visualizations
def plot_distributions(df):
    """
    Plots distributions of key numerical features.
    """
    numerical_cols = ["trip_distance", "fare_amount", "PU_freq", "DO_freq"]
    for col in numerical_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True, bins=30, color="blue")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()

def plot_correlations(df):
    """
    Plots a heatmap of feature correlations.
    """
    plt.figure(figsize=(10, 8))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

def plot_boxplots(df):
    """
    Plots boxplots for numerical features grouped by categorical features.
    """
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="rush_hour", y="fare_amount", data=df)
    plt.title("Fare Amount vs Rush Hour")
    plt.xlabel("Rush Hour")
    plt.ylabel("Fare Amount")
    plt.show()

# Run EDA
plot_distributions(df)
plot_correlations(df)
plot_boxplots(df)
