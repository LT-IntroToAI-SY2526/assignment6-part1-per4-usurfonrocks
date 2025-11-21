"""
Assignment 6 Part 1: Student Performance Prediction
Name: _______________
Date: _______________

This assignment predicts student test scores based on hours studied.
Complete all the functions below following the in-class ice cream example.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def load_and_explore_data(filename):
    """
    Load the student scores data and explore it
    
    Args:
        filename: name of the CSV file to load
    
    Returns:
        pandas DataFrame containing the data
    """
    # Load the data
    data = pd.read_csv(filename)

    print("=== Student Scores Data ===")
    print(f"\nFirst 5 rows:")
    print(data.head())

    print(f"\nDataset shape: {data.shape[0]} rows, {data.shape[1]} columns")

    print(f"\nBasic statistics:")
    print(data.describe())

    return data


def create_scatter_plot(data):
    """
    Create a scatter plot to visualize the relationship between hours studied and scores
    
    Args:
        data: pandas DataFrame with Hours and Scores columns
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Hours'], data['Scores'], color='purple', alpha=0.6)
    plt.xlabel('Hours Studied', fontsize=12)
    plt.ylabel('Test Score', fontsize=12)
    plt.title('Student Test Scores vs Hours Studied', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig('scatter_plot.png', dpi=300, bbox_inches='tight')
    print("\n✓ Scatter plot saved as 'scatter_plot.png'")
    plt.show()


def split_data(data):
    """
    Split data into features (X) and target (y), then into training and testing sets
    
    Args:
        data: pandas DataFrame with Hours and Scores columns
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Separate features (X) and target (y)
    X = data[['Hours']]
    y = data['Scores']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"\n=== Model Training Complete ===")
    print(f"Slope (coefficient): {model.coef_[0]:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"\nEquation: Score = {model.coef_[0]:.2f} × Hours + {model.intercept_:.2f}")

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
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    print(f"\n=== Model Performance ===")
    print(f"R² Score: {r2:.4f}")
    print(f"  → Interpretation: The model explains {r2*100:.2f}% of the variance in scores")

    print(f"\nMean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"  → Interpretation: On average, predictions are off by {rmse:.2f} points")

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

    plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Training Data')
    plt.scatter(X_test, y_test, color='green', alpha=0.7, label='Test Data (Actual)')
    plt.scatter(X_test, predictions, color='red', alpha=0.7, label='Predictions', marker='x', s=100)

    # Line of best fit
    X_range = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
    y_range = model.predict(X_range)
    plt.plot(X_range, y_range, color='black', linewidth=2, label='Line of Best Fit')

    plt.xlabel('Hours Studied', fontsize=12)
    plt.ylabel('Test Score', fontsize=12)
    plt.title('Linear Regression: Student Score Prediction', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('predictions_plot.png', dpi=300, bbox_inches='tight')
    print("\n✓ Predictions plot saved as 'predictions_plot.png'")
    plt.show()


def make_prediction(model, hours):
    """
    Make a prediction for a specific number of hours studied
    
    Args:
        model: trained LinearRegression model
        hours: number of hours to predict score for
    
    Returns:
        predicted test score
    """
    hours_array = np.array([[hours]])
    predicted_score = model.predict(hours_array)[0]

    print(f"\n=== New Prediction ===")
    print(f"If a student studies {hours} hours, predicted score: {predicted_score:.2f}")

    return predicted_score


if __name__ == "__main__":
    print("=" * 70)
    print("STUDENT PERFORMANCE PREDICTION - YOUR ASSIGNMENT")
    print("=" * 70)
    
    # Step 1: Load and explore the data
    data = load_and_explore_data('student_scores.csv')
    
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
    make_prediction(model, 7)
    
    print("\n" + "=" * 70)
    print("✓ Assignment complete! Check your saved plots.")
    print("Don't forget to complete a6_part1_writeup.md!")
    print("=" * 70) 
