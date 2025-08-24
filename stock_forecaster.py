# stock_forecaster.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine Learning & Forecasting
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from prophet import Prophet

import warnings
warnings.filterwarnings('ignore')



# stock_forecaster.py (continued)
@st.cache_data(ttl=3600) # Cache data for 1 hour to avoid reloading on every interaction
def load_data(ticker, start_date, end_date):
    """
    Fetches stock data from Yahoo Finance.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True) # Make 'Date' a column
    return data




# stock_forecaster.py (continued)
def create_lstm_dataset(dataset, time_step=60):
    """
    Creates the supervised learning dataset for an LSTM model.
    X_train: Past 'time_step' days of data
    y_train: The next day's closing price
    """
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i-time_step:i, 0]) # 0 is the column index (we'll only use 'Close')
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """
    Defines the architecture of the LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1)) # Output layer predicts the next closing price
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model



# stock_forecaster.py (continued)
def forecast_prophet(data, periods=30):
    """
    Forecasts using Facebook's Prophet model.
    Requires dataframe with columns 'ds' (datetime) and 'y' (value).
    """
    df_prophet = data[['Date', 'Close']].copy()
    df_prophet.columns = ['ds', 'y'] # Rename for Prophet

    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast, model

def forecast_lstm(data, time_step=60, forecast_days=30):
    """
    Forecasts using an LSTM model.
    """
    # Use only the 'Close' price
    dataset = data['Close'].values
    dataset = dataset.reshape(-1, 1)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    # Create the training dataset
    training_size = int(len(scaled_data) * 0.8) # Use 80% for training
    train_data = scaled_data[0:training_size, :]
    test_data = scaled_data[training_size - time_step:, :]

    X_train, y_train = create_lstm_dataset(train_data, time_step)

    # Reshape X_train to 3D for LSTM [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # Build and train the model
    model = build_lstm_model((X_train.shape[1], 1))
    history = model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=0) # Set verbose=1 to see training

    # Prepare the test data for prediction
    inputs = scaled_data[len(scaled_data) - len(test_data) - time_step:]
    inputs = inputs.reshape(-1,1)
    X_test, _ = create_lstm_dataset(inputs, time_step)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Predict on the test set
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions) # Reverse scaling

    # ----- FORECASTING FUTURE ----- #
    # Start with the last `time_step` from the original data
    future_forecast = []
    current_batch = scaled_data[-time_step:]
    current_batch = current_batch.reshape(1, time_step, 1)

    # Predict `forecast_days` into the future
    for i in range(forecast_days):
        next_pred = model.predict(current_batch)[0]
        future_forecast.append(next_pred)
        # Update the batch: remove first element, append the prediction
        current_batch = np.append(current_batch[:,1:,:], [[next_pred]], axis=1)

    future_forecast = scaler.inverse_transform(future_forecast)

    # Create dates for the forecast period
    last_date = data['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date, periods=forecast_days+1, freq='B')[1:] # Business days

    forecast_df = pd.DataFrame({'Date': future_dates, 'LSTM_Forecast': future_forecast.flatten()})
    return forecast_df, predictions, model




# stock_forecaster.py (continued)
def main():
    st.set_page_config(page_title="Stock Forecast Pro", page_icon="📈", layout="wide")
    st.title("📈 Stock Price Forecaster")
    st.markdown("""
    This tool uses historical data to forecast stock prices using **Facebook's Prophet** and **LSTM Neural Networks**.
    **Disclaimer:** This is for educational and demonstration purposes only. **Not financial advice.**
    """)

    # Sidebar for user input
    st.sidebar.header("User Input")
    ticker = st.sidebar.text_input("Stock Ticker Symbol (e.g., AAPL, TSLA, GOOGL):", "AAPL").upper()
    start_date = st.sidebar.date_input("Start Date:", pd.to_datetime("2018-01-01"))
    end_date = st.sidebar.date_input("End Date:", pd.to_datetime("today"))

    # Model selection and forecast horizon
    forecast_horizon = st.sidebar.slider("Forecast Horizon (Days):", 30, 180, 60)
    models_to_use = st.sidebar.multiselect(
        "Select Models to Run:",
        ["Prophet", "LSTM"],
        default=["Prophet", "LSTM"]
    )

    # Load data
    data_load_state = st.sidebar.text('Loading data...')
    df = load_data(ticker, start_date, end_date)
    data_load_state.text('Loading data... done!')

    if df.empty:
        st.error("No data found for that ticker symbol and date range. Please try again.")
        return

    # Display raw data
    st.subheader(f"Raw Data for {ticker}")
    st.dataframe(df.tail(), use_container_width=True)

    # Plot raw data
    fig_raw = go.Figure()
    fig_raw.add_trace(go.Candlestick(x=df['Date'],
                    open=df['Open'], high=df['High'],
                    low=df['Low'], close=df['Close'],
                    name='Market Data'))
    fig_raw.update_layout(title=f"{ticker} Share Price", yaxis_title='Price (USD)')
    st.plotly_chart(fig_raw, use_container_width=True)

    # Initialize forecast dataframes
    prophet_forecast_df = None
    lstm_forecast_df = None
    lstm_test_predictions = None

    # Run selected models
    if "Prophet" in models_to_use:
        with st.spinner('Training Prophet model... (This may take a minute)'):
            prophet_forecast_df, prophet_model = forecast_prophet(df, periods=forecast_horizon)

    if "LSTM" in models_to_use:
        with st.spinner('Training LSTM model... (This will take a few minutes)'):
            lstm_forecast_df, lstm_test_predictions, lstm_model = forecast_lstm(df, forecast_days=forecast_horizon)

    # Plot forecasts
    st.subheader("Forecast Results")
    fig_forecast = go.Figure()

    # Plot historical data
    fig_forecast.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Historical Close', line=dict(color='blue')))

    # Plot Prophet forecast
    if prophet_forecast_df is not None:
        fig_forecast.add_trace(go.Scatter(x=prophet_forecast_df['ds'], y=prophet_forecast_df['yhat'], mode='lines', name='Prophet Forecast', line=dict(color='green', dash='dash')))
        # Add uncertainty interval
        fig_forecast.add_trace(go.Scatter(x=prophet_forecast_df['ds'], y=prophet_forecast_df['yhat_upper'], fill=None, mode='lines', line=dict(width=0), showlegend=False))
        fig_forecast.add_trace(go.Scatter(x=prophet_forecast_df['ds'], y=prophet_forecast_df['yhat_lower'], fill='tonexty', mode='lines', line=dict(width=0), name='Prophet Uncertainty', opacity=0.3))

    # Plot LSTM forecast
    if lstm_forecast_df is not None:
        fig_forecast.add_trace(go.Scatter(x=lstm_forecast_df['Date'], y=lstm_forecast_df['LSTM_Forecast'], mode='lines', name='LSTM Forecast', line=dict(color='red')))

    fig_forecast.update_layout(title=f"{ticker} Price Forecast", yaxis_title='Price (USD)', xaxis_title='Date')
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Model Evaluation Section (Optional but impressive)
    st.subheader("Model Evaluation")
    if lstm_test_predictions is not None:
        # Create test dataset for evaluation
        train_size = int(len(df) * 0.8)
        test_df = df[train_size:].copy()
        test_df = test_df.reset_index(drop=True)

        # Ensure we only compare points we have predictions for
        min_length = min(len(test_df), len(lstm_test_predictions))
        test_df = test_df.iloc[:min_length]
        lstm_test_predictions = lstm_test_predictions[:min_length]

        # Calculate error metrics
        mae = mean_absolute_error(test_df['Close'], lstm_test_predictions)
        rmse = np.sqrt(mean_squared_error(test_df['Close'], lstm_test_predictions))

        col1, col2 = st.columns(2)
        col1.metric("MAE (Test Set)", f"${mae:.2f}")
        col2.metric("RMSE (Test Set)", f"${rmse:.2f}")

        # Plot LSTM predictions on test set
        fig_test = go.Figure()
        fig_test.add_trace(go.Scatter(x=test_df['Date'], y=test_df['Close'], mode='lines', name='Actual Test Price'))
        fig_test.add_trace(go.Scatter(x=test_df['Date'], y=lstm_test_predictions.flatten(), mode='lines', name='LSTM Prediction (Test)'))
        fig_test.update_layout(title="LSTM Model Performance on Test Data (Holdout Set)")
        st.plotly_chart(fig_test, use_container_width=True)

if __name__ == "__main__":
    main()


