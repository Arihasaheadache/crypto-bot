from keras.models import load_model
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np

CRYPTOS = ["BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD", "DOGE-USD"]
datafag = {}
models = {coin: load_model(f"models/{coin}.keras") for coin in CRYPTOS}


for crypto in CRYPTOS:
    lb = 48
    data = yf.download(crypto, period="3d", interval="1h")['Close'].dropna()

    if len(data) < lb:
        print("Not enough data")
        exit()

    lookback = 48  # Last 48 hours for input
    crypto_data = data
    if len(crypto_data) < lookback:
        print("Not enough historical data for prediction.")
        exit()

    scaler = MinMaxScaler(feature_range=(0, 1))
    historical_prices = np.array(crypto_data).reshape(-1, 1)
    scaler.fit(historical_prices)

    recent_data = np.array(crypto_data[-lookback:]).reshape(-1, 1)
    scaled_input = scaler.transform(recent_data).reshape(1, lookback, 1)

    indiv_pred_scaled = models[crypto].predict(scaled_input, verbose=0)
    indiv_pred = scaler.inverse_transform(indiv_pred_scaled.reshape(-1, 1))[0, 0]

    actual_price = (crypto_data.iloc[-1]).iloc[-1]

    price = f"{actual_price:,.2f}"
    prediction = f"{indiv_pred:,.2f}"

    datafag[crypto] = [price, prediction]

print(datafag)