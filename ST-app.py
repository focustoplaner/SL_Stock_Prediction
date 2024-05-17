import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def fetch_data(ticker_symbol, start_date, end_date):
    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    return data

def plot_close_price(data, line_color, fill_color, fill_area, show_grid):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['Close'], color=line_color)
    if fill_area:
        ax.fill_between(data.index, data['Close'], color=fill_color, alpha=0.3)
    if show_grid:
        ax.grid(True)
    ax.set_title('Close Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    st.pyplot(fig)

def train_linear_regression(data):
    X = np.arange(len(data)).reshape(-1, 1)
    y = data['Close'].values

    model = LinearRegression()
    model.fit(X, y)

    return model

def plot_scatter_and_prediction(data, model, scatter_color):
    X = np.arange(len(data)).reshape(-1, 1)
    y = data['Close'].values

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color=scatter_color, label='Actual')
    ax.plot(X, model.predict(X), color='red', label='Predicted')
    ax.set_title('Actual vs Predicted Close Price')
    ax.set_xlabel('Day')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

def calculate_model_error(model, data):
    X = np.arange(len(data)).reshape(-1, 1)
    y_actual = data['Close'].values
    y_pred = model.predict(X)
    mae = mean_absolute_error(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    return mae, mse

def main():
    st.title('Stock Price Prediction')

    ticker_symbol = st.text_input('Enter Ticker Symbol (e.g., NDX01.BK):', 'NDX01.BK')
    start_date = st.date_input('Start Date:', pd.to_datetime('2023-01-01'))
    end_date = st.date_input('End Date:', pd.to_datetime('today'))

    data = fetch_data(ticker_symbol, start_date, end_date)

    if not data.empty:
        st.write('**Data Overview:**')
        st.write(data.tail())

        line_color = st.color_picker('Choose Line Color', '#444CFF')
        scatter_color = st.color_picker('Choose Scatter Plot Color', '#444CFF')
        fill_area = st.checkbox('Fill area below graph', False)
        fill_color = st.color_picker('Choose Fill Color', '#AAF5FF') if fill_area else '#FFFFFF'
        show_grid = st.checkbox('Show Grid', False)

        st.write('**Close Price Graph:**')
        plot_close_price(data, line_color, fill_color, fill_area, show_grid)

        model = train_linear_regression(data)

        st.write('**Scatter Plot and Prediction:**')
        plot_scatter_and_prediction(data, model, scatter_color)

        # Calculate and display model error
        mae, mse = calculate_model_error(model, data)
        st.write('**Model Error:**')
        st.write(f"Mean Absolute Error (MAE): {mae}")
        st.write(f"Mean Squared Error (MSE): {mse}")

        # Predict next day's close price using linear regression
        next_day = pd.to_datetime(data.index[-1]) + pd.DateOffset(1)
        next_day_index = len(data) + 1
        predicted_price = model.predict([[next_day_index]])
        st.write('**Prediction for Next Day Close Price:**')
        st.write(f"Date: {next_day.strftime('%Y-%m-%d')}")
        st.write(f"Predicted Close Price: {predicted_price[0]}")
    else:
        st.write("No data available for the specified ticker symbol and date range.")

if __name__ == '__main__':
    main()
