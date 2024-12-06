import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sqlalchemy import create_engine
import mysql.connector

def load_dataset():
    # Dataset from Kaggle (Daily Weather Dataset)
    url = "https://github.com/rahiltrivedi/python_first/blob/main/DailyDelhiClimateTrain.csv"
    data = pd.read_csv(url, parse_dates=['date'], index_col='date')
    data = data[['temperature']]  # Focus on temperature for forecasting
    print("Dataset Loaded Successfully!")
    print(data.head())
    return data


def preprocess_data(data):
    # Handle missing values
    data = data.dropna()

    # Normalize temperature data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    return scaled_data, scaler

def create_time_series_dataset(data, time_steps=30):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps, 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def forecast(model, data, time_steps):
    last_time_step_data = data[-time_steps:]
    predictions = []
    for _ in range(7):  # Forecast next 7 days
        input_data = last_time_step_data.reshape(1, -1, 1)
        prediction = model.predict(input_data)[0][0]
        predictions.append(prediction)
        last_time_step_data = np.append(last_time_step_data[1:], prediction)
    return predictions

    def save_to_database(dates, values, db_config):
        engine = create_engine(
            f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}")
        results_df = pd.DataFrame({'date': dates, 'temperature': values})
        results_df.to_sql('forecast_results', con=engine, if_exists='replace', index=False)
        print("Forecasted results saved to database!")


if __name__ == "__main__":
    # MySQL Configuration
    db_config = {
        'user': 'root',
        'password': 'password',
        'host': 'localhost',
        'database': 'weather_forecasting_db'
    }

    # Step 1: Load Dataset
    data = load_dataset()

    # Step 2: Preprocess Data
    scaled_data, scaler = preprocess_data(data)
    time_steps = 30
    X, y = create_time_series_dataset(scaled_data, time_steps)

    # Step 3: Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape data for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Step 4: Build and Train Model
    model = build_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

    # Step 5: Forecast Future Data
    predictions = forecast(model, scaled_data, time_steps)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    # Step 6: Save Forecast to MySQL
    future_dates = pd.date_range(start=data.index[-1] + pd.DateOffset(1), periods=7)
    save_to_database(future_dates, predictions, db_config)

    # Step 7: Visualize Results
    plt.figure(figsize=(14, 7))
    plt.plot(data.index[-100:], data['temperature'].iloc[-100:], label='Historical Data')
    plt.plot(future_dates, predictions, label='Forecasted Data', linestyle='--')
    plt.title('Weather Forecasting')
    plt.xlabel('Date')
    plt.ylabel('Temperature')
    plt.legend()
    plt.show()
