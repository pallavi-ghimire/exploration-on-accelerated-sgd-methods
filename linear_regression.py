"""
This file implements Linear Regression in order to set up benchmark scores for the thesis to build upon.
Scores are compared with the paper by Singh, Gurjeet (2022) - Machine Learning Models in Stock Market Prediction

Implementation of Linear Regression is done in 2 forms:
1. Using the function LinearRegression() from sklearn.linear_model
2. Implementing in the form of Close = w1*Open + w2*High + w3*Low + intercept, inspired from Tsay, R.S. (2010) -
Analysis of Financial Time Series, which is further combined with SGD
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
from sklearn.model_selection import KFold

df = pd.read_csv('dataset/NIFTY_50.csv')

# Convert Date column to datetime, then reset index if not already sorted by date
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

scaler = StandardScaler()
df[['Open', 'High', 'Low']] = scaler.fit_transform(df[['Open', 'High', 'Low']])

# Define features and target
X = ['Open', 'High', 'Low']
y = 'Close'

# Divide data into chunks
no_of_chunks = 4
n = len(df)
chunk_size = n // no_of_chunks
split_dataset = []


# Create subsets for data
def create_subsets_for_data():
    print(f"\nCreating Subsets for data")

    for i in range(no_of_chunks):
        # Calculate start row index of each chunk (e.g. if i = 2, chunk_size = 1000, then start = 2000)
        # start = i * chunk_size
        start = 0
        # Calculate end row index of each chunk (e.g. if i = 2, chunk_size = 1000, then start = 3000)
        end = (i + 1) * chunk_size if i < (
                no_of_chunks - 1) else n  # make sure to include the remainder in the last chunk
        # Extract chunk of data using row slicing
        print("for subset ", i + 1, ", ", start, " = start size, ", end, " = end size")
        subset = df.iloc[start:end]
        # print(subset.head(1))

        # Split 80% train, 20% test within the chunk
        train_end = int(len(subset) * 0.8)

        X_train = subset[X].iloc[:train_end]
        y_train = subset[y].iloc[:train_end]
        X_test = subset[X].iloc[train_end:]
        y_test = subset[y].iloc[train_end:]

        split_dataset.append((X_train, X_test, y_train, y_test))


"""
Implement linear regression model to tally benchmark scores from inspiration paper.
LinearRegression() uses OLS method.
"""


def linear_regression_using_sklearn(splits):
    print(f"\nRunning LinearRegression() to tally benchmark scores from inspiration paper")

    for idx, (X_train, X_test, y_train, y_test) in enumerate(splits):
        # Initialize and train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse = mean_squared_error(y_test, y_pred)

        # Print model coefficients and performance
        # print("Model Coefficients:", dict(zip(X, model.coef_)))  # for getting the model coefficients
        print("\nCoefficients, Intercept and MSE for subset", idx + 1)
        for feature, coefficient in zip(X,
                                        model.coef_.flatten()):  # for mapping the coefficients with respective features
            print(f"{feature}: {coefficient:.4f}")
        print("Intercept:", model.intercept_)
        print("Mean Squared Error for Training Data:", mse_train)
        print("Mean Squared Error for Testing Data:", mse)


# K-fold cv. Will be using no_of_chunks as k.
def cross_validate_on_training_data(splits, n_splits):
    print(f"\nRunning {n_splits}-Fold Cross-Validation on Training Data")

    # Keep X_train and y_train to apply Cross-Validation, ignore X_test and y_test
    for idx, (X_train, _, y_train, _) in enumerate(splits):
        # Set up n_splits as number of folds
        kf = KFold(n_splits=n_splits, shuffle=False)  # shuffle=False to preserve time order, since data is time series
        mse_list = []

        # Split X_train data into training and validation indices. Split function gives index numbers,
        # and doesn't directly split data
        for fold_idx, (train_index, val_index) in enumerate(kf.split(X_train)):
            X_cv_train, X_cv_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_cv_train, y_cv_val = y_train.iloc[train_index], y_train.iloc[val_index]

            # Initialize and train the Linear Regression model
            model = LinearRegression()
            model.fit(X_cv_train, y_cv_train)

            # Make prediction
            y_cv_pred = model.predict(X_cv_val)
            mse = mean_squared_error(y_cv_val, y_cv_pred)
            mse_list.append(mse)

        avg_mse = np.mean(mse_list)
        print(f"Subset {idx + 1} - Avg MSE from {n_splits}-Fold CV on Training Data: {avg_mse:.4f}")  # up to 4 float pt


# Implement linear regression with sgd to create benchmark scores for thesis
def linear_regression_using_custom_sgd(splits, learning_rate=0.001, epochs=30000):
    print(f"\nRunning Custom SGD-Based Linear Regression to tally benchmark scores from inspiration paper")
    print("learning rate:", learning_rate, "epochs:", epochs)

    for idx, (X_train, X_test, y_train, y_test) in enumerate(splits):
        # Df has data in the form of numpy arrays under values. Convert them to numpy arrays
        X_train_np = X_train.values
        y_train_np = y_train.values
        X_test_np = X_test.values
        y_test_np = y_test.values

        # set dimension of n_samples and n_features as of X_train_np
        n_samples, n_features = X_train_np.shape

        # Initialize weights and bias
        weights = np.zeros(n_features)
        bias = 0

        # SGD training
        for epoch in range(epochs):  # for each epoch
            for i in range(n_samples):  # for each sample
                xi = X_train_np[i]
                yi = y_train_np[i]

                # Prediction using Linear Regression
                y_pred = np.dot(xi, weights) + bias

                # Check if prediction is nan or inf
                if np.isnan(y_pred) or np.isinf(y_pred):
                    print(f"Found bad y_pred at epoch {epoch}, sample {i}")
                    print("xi:", xi)
                    print("weights:", weights)
                    print("bias:", bias)
                    break  # skip this bad sample

                # Error
                error = 2 * (y_pred - yi)

                # Update rule
                weights -= learning_rate * error * xi
                bias -= learning_rate * error

        # After training, predict
        # fit to linear regression the training data, with the updated weights
        y_train_pred = np.dot(X_train_np, weights) + bias
        # fit to linear regression the test data, with the updated weights
        y_pred = np.dot(X_test_np, weights) + bias

        # Evaluate using MSE
        mse_train = mean_squared_error(y_train_np, y_train_pred)
        mse_test = mean_squared_error(y_test_np, y_pred)

        # Print results
        print("\nSubset", idx + 1)
        for feature, coefficient in zip(X, weights):
            print(f"{feature}: {coefficient:.4f}")
        print(f"Intercept (Bias): {bias:.4f}")
        print(f"Mean Squared Error for Training Data: {mse_train:.4f}")
        print(f"Mean Squared Error for Testing Data: {mse_test:.4f}")


# Call the functions
create_subsets_for_data()
# linear_regression_using_sklearn(split_dataset)
# cross_validate_on_training_data(split_dataset, no_of_chunks)
linear_regression_using_custom_sgd(split_dataset, 0.001, 30000)

print("\nMean Close Value:", df['Close'].mean(), "and highest/lowest value:", df['Close'].max(), "&", df['Close'].min())
