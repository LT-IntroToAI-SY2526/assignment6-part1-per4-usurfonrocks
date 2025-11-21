"""
Ice Cream Sales Prediction - In-Class Example
Mr. Berg - Introduction to AI

This example demonstrates linear regression by predicting ice cream sales based on temperature.
We'll walk through each step together in class.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def load_and_explore_data(filename):
    """
    Load the ice cream sales data and explore it
    
    Args:
        filename: name of the CSV file to load
    
    Returns:
        pandas DataFrame containing the data
    """
    # Load the data
    data = pd.read_csv(filename)
    
    print("=== Ice Cream Sales Data ===")
    print(f"\nFirst 5 rows:")
    print(data.head())
    
    print(f"\nDataset shape: {data.shape[0]} rows, {data.shape[1]} columns")
    
    print(f"\nBasic statistics:")
    print(data.describe())
    
    return data


def create_scatter_plot(data):
    """
    Create a scatter plot to visualize the relationship between temperature and sales
    
    Args:
        data: pandas DataFrame with Temperature and Sales columns
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Temperature'], data['Sales'], color='blue', alpha=0.6)
    plt.xlabel('Temperature (°F)', fontsize=12)
    plt.ylabel('Ice Cream Sales ($)', fontsize=12)
    plt.title('Ice Cream Sales vs Temperature', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig('ice_cream_scatter.png', dpi=300, bbox_inches='tight')
    print("\n✓ Scatter plot saved as 'ice_cream_scatter.png'")
    plt.show()


def split_data(data):
    """
    Split data into features (X) and target (y), then into training and testing sets
    
    Args:
        data: pandas DataFrame with Temperature and Sales columns
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Separate features (X) and target (y)
    X = data[['Temperature']]  # Features - note the double brackets to keep it as DataFrame
    y = data['Sales']           # Target variable
    
    # Split into training (80%) and testing (20%) sets
    # random_state=42 ensures we get the same split every time (reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n=== Data Split ===")
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Create and train a linear regression model
    
    Args:
        X_train: training features
        y_train: training target values
    
    Returns:
        trained LinearRegression model
    """
    # Create the model
    model = LinearRegression()
    
    # Train the model (this is where the "learning" happens!)
    model.fit(X_train, y_train)
    
    print(f"\n=== Model Training Complete ===")
    print(f"Slope (coefficient): {model.coef_[0]:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"\nEquation: Sales = {model.coef_[0]:.2f} × Temperature + {model.intercept_:.2f}")
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model's performance on test data
    
    Args:
        model: trained LinearRegression model
        X_test: testing features
        y_test: testing target values
    
    Returns:
        predictions array
    """
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    print(f"\n=== Model Performance ===")
    print(f"R² Score: {r2:.4f}")
    print(f"  → Interpretation: The model explains {r2*100:.2f}% of the variance in sales")
    
    print(f"\nMean Squared Error: ${mse:.2f}")
    print(f"Root Mean Squared Error: ${rmse:.2f}")
    print(f"  → Interpretation: On average, predictions are off by ${rmse:.2f}")
    
    return predictions


def visualize_results(X_train, y_train, X_test, y_test, predictions, model):
    """
    Visualize the model's predictions against actual values
    
    Args:
        X_train: training features
        y_train: training target values
        X_test: testing features
        y_test: testing target values
        predictions: model predictions on test set
        model: trained model (to plot line of best fit)
    """
    plt.figure(figsize=(12, 6))
    
    # Plot training data
    plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Training Data')
    
    # Plot test data
    plt.scatter(X_test, y_test, color='green', alpha=0.7, label='Test Data (Actual)')
    
    # Plot predictions
    plt.scatter(X_test, predictions, color='red', alpha=0.7, label='Predictions', marker='x', s=100)
    
    # Plot line of best fit
    X_range = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
    y_range = model.predict(X_range)
    plt.plot(X_range, y_range, color='black', linewidth=2, label='Line of Best Fit')
    
    plt.xlabel('Temperature (°F)', fontsize=12)
    plt.ylabel('Ice Cream Sales ($)', fontsize=12)
    plt.title('Linear Regression: Ice Cream Sales Prediction', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('ice_cream_predictions.png', dpi=300, bbox_inches='tight')
    print("\n✓ Predictions plot saved as 'ice_cream_predictions.png'")
    plt.show()


def make_prediction(model, temperature):
    """
    Make a prediction for a specific temperature
    
    Args:
        model: trained LinearRegression model
        temperature: temperature value to predict sales for
    
    Returns:
        predicted sales value
    """
    # Reshape temperature into the format the model expects
    temp_array = np.array([[temperature]])
    predicted_sales = model.predict(temp_array)[0]
    
    print(f"\n=== New Prediction ===")
    print(f"If temperature is {temperature}°F, predicted sales: ${predicted_sales:.2f}")
    
    return predicted_sales


if __name__ == "__main__":
    print("=" * 60)
    print("ICE CREAM SALES PREDICTION - LINEAR REGRESSION EXAMPLE")
    print("=" * 60)
    
    # Step 1: Load and explore the data
    data = load_and_explore_data('ice_cream_sales.csv')
    
    # Step 2: Visualize the relationship
    create_scatter_plot(data)
    
    # Step 3: Split the data
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Step 4: Train the model
    model = train_model(X_train, y_train)
    
    # Step 5: Evaluate the model
    predictions = evaluate_model(model, X_test, y_test)
    
    # Step 6: Visualize results
    visualize_results(X_train, y_train, X_test, y_test, predictions, model)
    
    # Step 7: Make a new prediction
    make_prediction(model, 90)
    
    print("\n" + "=" * 60)
    print("✓ Example complete! Check out the saved plots.")
    print("=" * 60)
