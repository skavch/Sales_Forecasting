import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from matplotlib import pyplot as plt

# Function to preprocess the dataset
def preprocess_data(df, target_col):
    df = df.copy()
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)
    df = df.dropna()
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    df = pd.DataFrame(df_scaled, index=df.index, columns=df.columns)
    return df, scaler

# Function to create sequences for LSTM/GRU/XGBoost
def create_sequences(data, target_col, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data.iloc[i:i + n_steps].values)
        y.append(data.iloc[i + n_steps][target_col])
    return np.array(X), np.array(y)

# Function to train LSTM model with early stopping
def train_lstm_model(X, y):
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X, y, epochs=300, verbose=0, callbacks=[early_stopping])
    return model

# Function to train GRU model with early stopping
def train_gru_model(X, y):
    model = Sequential()
    model.add(GRU(100, activation='relu', input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X, y, epochs=300, verbose=0, callbacks=[early_stopping])
    return model

# Function to train XGBoost model
def train_xgboost_model(X, y):
    X_reshaped = X.reshape(X.shape[0], -1)  # Reshape for XGBoost
    model = XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.01, max_depth=5)
    model.fit(X_reshaped, y)
    return model

# Function to train ARIMA model with default parameters
def train_arima_model(data):
    model = ARIMA(data, order=(1, 1, 1))
    model_fit = model.fit()
    return model_fit

# Function to train SARIMA model with default parameters
def train_sarima_model(data):
    model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)
    return model_fit

# Function to evaluate models
def evaluate_models(models, X, y, df, scaler, target_col, n_steps):
    results = {}
    for name, model in models.items():
        if name in ['LSTM', 'GRU']:
            y_pred = model.predict(X)
        elif name == 'XGBoost':
            y_pred = model.predict(X.reshape(X.shape[0], -1))
        elif name in ['ARIMA', 'SARIMA']:
            y_pred = model.predict(start=0, end=len(y)-1)
        
        y_true = scaler.inverse_transform(y.reshape(-1, 1))
        y_pred = scaler.inverse_transform(np.array(y_pred).reshape(-1, 1))
        mse = mean_squared_error(y_true, y_pred)
        results[name] = mse
    return results

# Function to forecast future values
def forecast_best_model(best_model, model_name, data, scaler, n_steps, n_months):
    if model_name in ['LSTM', 'GRU']:
        # For LSTM/GRU
        predictions = []
        input_seq = data[-n_steps:]
        input_seq = input_seq.reshape((1, n_steps, 1))
        for _ in range(n_months):
            pred = best_model.predict(input_seq)[0, 0]
            predictions.append(pred)
            new_input = np.array([pred])
            input_seq = np.append(input_seq[:, 1:, :], new_input.reshape((1, 1, 1)), axis=1)
        return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    elif model_name == 'XGBoost':
        # For XGBoost
        input_seq = data[-n_steps:].reshape(1, -1)
        predictions = []
        for _ in range(n_months):
            pred = best_model.predict(input_seq)[0]
            predictions.append(pred)
            input_seq = np.append(input_seq[:, 1:], pred).reshape(1, -1)
        return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    elif model_name in ['ARIMA', 'SARIMA']:
        # For ARIMA/SARIMA
        forecast = best_model.forecast(steps=n_months)
        return scaler.inverse_transform(np.array(forecast).reshape(-1, 1))


# Streamlit UI
st.set_page_config(page_title="Skavch Sales Forecasting Engine", page_icon=":chart_with_upwards_trend:", layout="wide")

# Add an image to the header
st.image("bg1.jpg", use_column_width=True)  # Adjust the image path as necessary

st.title("Skavch Sales Forecasting Engine")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
target_col = st.text_input("Enter the target column name")
n_months = st.number_input("Enter the number of future months to forecast", min_value=1, max_value=36)
submit_button = st.button(label="Submit")

if submit_button:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if target_col in df.columns:
            df, scaler = preprocess_data(df, target_col)
            st.write("Data Preprocessed Successfully!")

            # Prepare data for models
            n_steps = 12
            X, y = create_sequences(df, target_col, n_steps)

            # Train models
            lstm_model = train_lstm_model(X, y)
            gru_model = train_gru_model(X, y)
            xgb_model = train_xgboost_model(X, y)
            arima_model = train_arima_model(df[target_col])
            sarima_model = train_sarima_model(df[target_col])

            # Evaluate models
            models = {'LSTM': lstm_model, 'GRU': gru_model, 'XGBoost': xgb_model, 'ARIMA': arima_model, 'SARIMA': sarima_model}
            results = evaluate_models(models, X, y, df, scaler, target_col, n_steps)

            # Display evaluation results
            st.subheader("Model Evaluation Results")
            for model_name, mse in results.items():
                st.write(f"{model_name} Model MSE: {mse:.4f}")

            # Select best model based on MSE
            best_model_name = min(results, key=results.get)
            best_model = models[best_model_name]
            st.write(f"Best Model: {best_model_name} with MSE: {results[best_model_name]:.4f}")

            # Forecast future values with the best model
            future_forecast = forecast_best_model(best_model, best_model_name, df[target_col].values, scaler, n_steps, n_months)

            # Plotting
            st.subheader("Forecast Results")
            plt.figure(figsize=(18, 9))
            plt.plot(df.index, scaler.inverse_transform(df[target_col].values.reshape(-1, 1)), label='Historical Data')
            future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=n_months, freq='M')
            plt.plot(future_dates, future_forecast, label='Forecasted Data', color='red')
            plt.xlabel('Date')
            plt.ylabel(target_col)
            plt.title('Time Series Forecast')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)
        else:
            st.error(f"Target column '{target_col}' not found in the dataset.")
    else:
        st.error("Please upload a CSV file.")