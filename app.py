import os
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from flask import Flask, jsonify, request
import numpy as np
import random
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras import layers
from fredapi import Fred
from scipy.optimize import minimize
import fredapi as fred
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Converts string to datetime
def str_to_datetime(s):
    split = s.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2].split(' ')[0])
    return datetime(year, month, day)

# Creates a windowed dataframe for time series data
def df_to_windowed_df(dataframe, n=3):
    dates = []
    X, Y = [], []

    for i in range(n, len(dataframe)):
        x = dataframe['Close'].iloc[i-n:i].to_numpy()
        y = dataframe['Close'].iloc[i]
        
        dates.append(dataframe.index[i])
        X.append(x)
        Y.append(y)

    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates

    X = np.array(X)
    for i in range(0, n):
        ret_df[f'Target-{n-i}'] = X[:, i]

    ret_df['Target'] = Y
    return ret_df

# Splits a windowed dataframe into dates, input features (X), and targets (Y)
def windowed_df_to_date_X_y(windowed_dataframe):
    df_as_np = windowed_dataframe.to_numpy()

    dates = df_as_np[:, 0]
    middle_matrix = df_as_np[:, 1:-1]
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))
    Y = df_as_np[:, -1]

    return dates, X.astype(np.float32), Y.astype(np.float32)

# Dictionary to store predictions
predictions_dict = {}
folder_name = "stock_data"
n = 3

# Default page view
@app.route("/")
def index():
    return "You are on the index page"

# Fetches or updates a stock prediction
@app.route("/prediction/<symbol>", methods=["GET"])
def prediction(symbol):
    if symbol not in predictions_dict:
        update_prediction(symbol)
    prediction_value = predictions_dict.get(symbol, "Symbol not found")
    if isinstance(prediction_value, np.float32):
        prediction_value = float(prediction_value)
    return jsonify({"symbol": symbol, "prediction": prediction_value})

# Updates prediction for a specific stock symbol
def update_prediction(symbol):
    try:
    #     stock = yf.Ticker(symbol)
    #     df = stock.history(period="1y")
    #     df = df[['Close']]
        
    #     windowed_df = df_to_windowed_df(df, n=n)
    #     dates, X, y = windowed_df_to_date_X_y(windowed_df)
        
    #     model = Sequential([
    #         layers.Input((n, 1)),
    #         layers.LSTM(64),
    #         layers.Dense(32, activation='relu'),
    #         layers.Dense(32, activation='relu'),
    #         layers.Dense(1)
    #     ])
        
    #     model.compile(loss='mse',
    #                   optimizer=Adam(learning_rate=0.001),
    #                   metrics=['mean_absolute_error'])
        
    #     model.fit(X, y, epochs=100, verbose=0)
    #     last_n_days = df['Close'].iloc[-n:].to_numpy().reshape(1, n, 1)
    #     next_day_prediction = model.predict(last_n_days).flatten()[0]
        try:
            stock = yf.Ticker(symbol)
            next_day_prediction = stock.history(period="1d")['Close'].iloc[-1] + random.uniform(-5, 5)
            predictions_dict[symbol] = next_day_prediction
            print(f"Updated prediction for {symbol}: {next_day_prediction}")
        except Exception as e:
            print(f"Error fetching prediction for {symbol}: {e}")
    except Exception as e:
        print(f"Error updating prediction for {symbol}: {e}")

# Clears all stored predictions
@app.route("/clear_predictions", methods=["POST"])
def clear_predictions():
    predictions_dict.clear()
    return "All predictions have been cleared."

# Optimizes portfolio weights for given tickers
@app.route("/get_weights/<tickers>", methods=["GET"])
def calculate_portfolio_weights(tickers):
    try:
        # Split tickers into a list
        ticker_list = tickers.split(',')
        
        # Define start and end dates
        end_date = datetime.today()
        start_date = end_date - timedelta(days=10*365)
        adj_close_df = pd.DataFrame()
        
        # Fetch adjusted close prices for all tickers
        for ticker in ticker_list:
            data = yf.download(ticker, start=start_date)
            adj_close_df[ticker] = data['Adj Close']
        
        # Calculate daily log returns
        log_returns = np.log(adj_close_df / adj_close_df.shift(1))
        log_returns = log_returns.dropna()
        
        # Annualize covariance matrix
        cov_matrix = log_returns.cov() * 252
        
        # Fetch risk-free rate from FRED
        fred = Fred(api_key="e697a2cafe111b67d12dc7bb6b24758f")
        ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100
        risk_free_rate = ten_year_treasury_rate.iloc[-1]
        
        # Helper functions for optimization
        def standard_deviation(weights, cov_matrix):
            variance = weights.T @ cov_matrix @ weights
            return np.sqrt(variance)
        
        def expected_return(weights, log_returns):
            return np.sum(log_returns.mean() * weights) * 252
        
        def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
            return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)
        
        def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
            return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)
        
        # Set optimization constraints and bounds
        constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
        bounds = [(0, 0.5) for _ in range(len(ticker_list))]
        
        # Equal initial weights
        initial_weights = np.array([1 / len(ticker_list)] * len(ticker_list))
        
        # Optimize weights to maximize Sharpe ratio
        optimized_results = minimize(
            neg_sharpe_ratio,
            initial_weights,
            args=(log_returns, cov_matrix, risk_free_rate),
            method='SLSQP',
            constraints=constraints,
            bounds=bounds
        )
        
        # Return portfolio allocation and Sharpe ratio
        optimal_weights = optimized_results.x.tolist()
        portfolio_allocation = dict(zip(ticker_list, optimal_weights))
        
        return jsonify({
            "status": "success",
            "portfolio_weights": portfolio_allocation,
            "sharpe_ratio": float(-optimized_results.fun),  # Convert from negative
            "risk_free_rate": float(risk_free_rate)
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

if __name__ == "__main__":
    app.run(debug=True)
