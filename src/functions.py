import datetime
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def find_cryptodata(symbol): # Find crypto data from Yahoo Finance
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=60)

    data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'),end=end_date.strftime('%Y-%m-%d'), interval="1h")

    return data[['Close']]

def preprocess_data(lb,data): # Fits data for individual crypto model's scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X,y = [], []

    for i in range(lb, len(scaled_data)-1):
        X.append(scaled_data[i-lb:i])
        y.append(scaled_data[i + 1])

    return np.array(X), np.array(y), scaler

def build_model(lb): # A fits-all model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lb, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def train_model(lb,ep,bs,symbol): # Training the individual model
    print(f"Training model: {symbol} \n")
    data = find_cryptodata(symbol)

    X, y, scaler = preprocess_data(lb,data)
    model = build_model(lb)
    history = model.fit(X, y, epochs=ep, batch_size=bs, verbose=1)
    
    model.save(f"{symbol}.keras")
    print(f"Model saved: {symbol} \n")

    return model, scaler, X, y

def eval_model(model, scaler, X, y): #Evaluating the model for an error percentage based on current data
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    actuals = scaler.inverse_transform(y)

    error = np.mean(np.abs(predictions - actuals) / actuals) * 100
    print(f"Error: {error:.2f}%")
    return error