import os
from flask import Flask, render_template, send_file
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

app = Flask(__name__)


@app.route('/')
def index():
    # Define the ticker symbol for the gold ETF
    ticker_symbol = 'GLD'

    # Fetch historical gold prices using yfinance without specifying a start date
    gold_data = yf.download(
        ticker_symbol, end=datetime.now().strftime('%Y-%m-%d'))

    # Feature engineering: Adding lag features for time series data
    gold_data['GoldPrice_Lag1'] = gold_data['Close'].shift(1)
    gold_data.dropna(inplace=True)

    # Classify the gold prices into "Up" or "Down" based on the next day's closing price
    gold_data['UpDown'] = np.where(
        gold_data['Close'].shift(-1) > gold_data['Close'], 1, 0)

    # Split data into training, validation, and testing sets for regression
    X_train_val_reg, X_test_reg, y_train_val_reg, y_test_reg = train_test_split(
        gold_data[['GoldPrice_Lag1']], gold_data['Close'], test_size=0.2, random_state=42
    )

    X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(
        X_train_val_reg, y_train_val_reg, test_size=0.25, random_state=42
    )

    # Split data into training, validation, and testing sets for classification
    X_train_val_class, X_test_class, y_train_val_class, y_test_class = train_test_split(
        gold_data[['GoldPrice_Lag1']], gold_data['UpDown'], test_size=0.2, random_state=42
    )

    X_train_class, X_val_class, y_train_class, y_val_class = train_test_split(
        X_train_val_class, y_train_val_class, test_size=0.25, random_state=42
    )

    # Train a linear regression model for regression
    reg_model = LinearRegression()
    reg_model.fit(X_train_reg, y_train_reg)

    # Train a random forest regressor for regression
    rf_regressor = RandomForestRegressor(random_state=42)
    rf_regressor.fit(X_train_reg, y_train_reg)

    # Create an ensemble model using a voting regressor
    ensemble_model = VotingRegressor(estimators=[
        ('linear_regression', reg_model),
        ('random_forest_regressor', rf_regressor)
    ])

    # Fit the ensemble model on the training data for regression
    ensemble_model.fit(X_train_reg, y_train_reg)

    # Make predictions on the validation set using the ensemble model for regression
    y_pred_val_ensemble_reg = ensemble_model.predict(X_val_reg)

    # Calculate residuals (prediction errors) on the validation set for regression
    residuals_val_ensemble_reg = y_val_reg - y_pred_val_ensemble_reg

    # Calculate volatility (standard deviation) of residuals for regression as a measure of risk
    volatility_val_ensemble_reg = np.std(residuals_val_ensemble_reg)

    # Plot actual vs predicted prices on the validation set for regression
    plt.figure(figsize=(10, 6))
    plt.scatter(X_val_reg, y_val_reg, color='black', label='Actual Prices')
    plt.plot(X_val_reg, y_pred_val_ensemble_reg, color='blue',
             linewidth=3, label='Predicted Prices (Regression)')
    plt.xlabel('Gold Price (lagged)')
    plt.ylabel('Gold Price')
    plt.title(
        f'Gold Price Prediction (Regression) for {ticker_symbol} on the Validation Set')
    plt.legend()
    plt.savefig('static/validation_plot_regression.png')

    # Make predictions on the test set using the ensemble model for regression
    y_pred_test_ensemble_reg = ensemble_model.predict(X_test_reg)

    # Evaluate the regression model on the test set
    mse_test_reg = mean_squared_error(y_test_reg, y_pred_test_ensemble_reg)

    # Calculate residuals (prediction errors) on the test set for regression
    residuals_test_reg = y_test_reg - y_pred_test_ensemble_reg

    # Calculate volatility (standard deviation) of residuals for regression as a measure of risk
    volatility_test_reg = np.std(residuals_test_reg)

    # Plot actual vs predicted prices on the test set for regression
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test_reg, y_test_reg, color='black', label='Actual Prices')
    plt.plot(X_test_reg, y_pred_test_ensemble_reg, color='blue',
             linewidth=3, label='Predicted Prices (Regression)')
    plt.xlabel('Gold Price (lagged)')
    plt.ylabel('Gold Price')
    plt.title(
        f'Gold Price Prediction (Regression) for {ticker_symbol} on the Test Set')
    plt.legend()
    plt.savefig('static/prediction_plot_regression.png')

    # Make predictions on the test set using the ensemble model for classification
    y_pred_test_ensemble_class = ensemble_model.predict(X_test_class)

    # Calculate accuracy on the test set for the classification model
    accuracy_test_class = accuracy_score(
        y_test_class, (y_pred_test_ensemble_class > 0.5).astype(int))

    # Make predictions on the historical data (backtesting) for regression
    gold_data['Predicted'] = ensemble_model.predict(
        gold_data[['GoldPrice_Lag1']])

    # Calculate residuals (prediction errors) on the historical data for regression
    residuals_reg = gold_data['Close'] - gold_data['Predicted']

    # Calculate volatility (standard deviation) of residuals for regression as a measure of risk
    volatility_reg = np.std(residuals_reg)

    # Evaluate the regression model on the test set for backtesting
    mse_test_backtesting = mean_squared_error(
        gold_data['Close'], ensemble_model.predict(gold_data[['GoldPrice_Lag1']]))

    # Download gold_data DataFrame as CSV
    gold_data.to_csv('static/gold_data.csv', index=False)

    return render_template('index.html', mse_test_reg=mse_test_reg,
                           volatility_test_reg=volatility_test_reg,
                           accuracy_test_class=accuracy_test_class,
                           mse_test_backtesting=mse_test_backtesting,
                           volatility_reg=volatility_reg,
                           volatility_val_ensemble_reg=volatility_val_ensemble_reg)


@app.route('/download_data')
def download_data():
    file_path = os.path.join(app.root_path, 'static', 'gold_data.csv')
    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
