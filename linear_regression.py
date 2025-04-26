# will be implementing in the form of
# log(Close) = w1*Open + w2*High + w3*Low + intercept
# and then converting back with exp()
# inspired from Tsay, R.S. (2010) – Analysis of Financial Time Series
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
from sklearn.model_selection import KFold

df = pd.read_csv('dataset/NIFTY_50.csv')

# Convert Date column to datetime, then reset index if not already sorted by date
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Define features and target
X = ['Open', 'High', 'Low']
y = 'Close'

# Divide data into chunks
no_of_chunks = 4
n = len(df)
chunk_size = n // no_of_chunks
split_dataset = []


# create subsets for data
def create_subsets_for_data():
    for i in range(no_of_chunks):
        # calculate start row index of each chunk (e.g. if i = 2, chunk_size = 1000, then start = 2000)
        # start = i * chunk_size
        start = 0
        # calculate end row index of each chunk (e.g. if i = 2, chunk_size = 1000, then start = 3000)
        end = (i+1) * chunk_size if i < (no_of_chunks-1) else n  # make sure to include the remainder in the last chunk
        # extract chunk of data using row slicing
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


# implement linear regression model to tally benchmark scores from inspiration paper
def linear_regression_using_sklearn(splits):
    for idx, (X_train, X_test, y_train, y_test) in enumerate(splits):
        # Initialize and train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)

        # Print model coefficients and performance
        # print("Model Coefficients:", dict(zip(X, model.coef_)))  # for getting the model coefficients
        print("\ncoefficients, intercept and MSE for subset", idx + 1)
        for feature, coefficient in zip(X,
                                        model.coef_.flatten()):  # for mapping the coefficients with respective features
            print(f"{feature}: {coefficient:.4f}")
        print("Intercept:", model.intercept_)
        print("Mean Squared Error:", mse)


# k-fold cv
def cross_validate_on_training_data(splits, n_splits):
    print(f"\nRunning {n_splits}-Fold Cross-Validation on Training Data...\n")

    for idx, (X_train, _, y_train, _) in enumerate(splits):
        kf = KFold(n_splits=n_splits, shuffle=False)  # shuffle=False to preserve time order
        mse_list = []

        for fold_idx, (train_index, val_index) in enumerate(kf.split(X_train)):
            X_cv_train, X_cv_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_cv_train, y_cv_val = y_train.iloc[train_index], y_train.iloc[val_index]

            model = LinearRegression()
            model.fit(X_cv_train, y_cv_train)
            y_cv_pred = model.predict(X_cv_val)
            mse = mean_squared_error(y_cv_val, y_cv_pred)
            mse_list.append(mse)

        avg_mse = np.mean(mse_list)
        print(f"Subset {idx + 1} - Avg MSE from {n_splits}-Fold CV on Training Data: {avg_mse:.4f}")

# implement linear regression with sgd to create benchmark scores for thesis


# call the functions
create_subsets_for_data()
cross_validate_on_training_data(split_dataset, no_of_chunks)
linear_regression_using_sklearn(split_dataset)
