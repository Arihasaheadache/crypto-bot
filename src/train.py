#&

import functions as fn
import os
os.makedirs("models", exist_ok=True)

CRYPTOS = ["BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD", "DOGE-USD"] # Current currencies added for popularity, you can add more if you want

# Parameters

LOOK_BACK = 48 # Lookback at 48 hours worth of crypto readings
EPOCHS = 40 # Number of epochs
BATCH_SIZE = 32 # Batch size

for symbol in CRYPTOS:
    best_model = None
    best_scaler = None
    best_error = float('inf') #Starting error at infinity

    while best_error > 2.0:
        model, scaler, X, y = fn.train_model(LOOK_BACK, EPOCHS, BATCH_SIZE, symbol)
        error = fn.eval_model(model, scaler, X, y)

        if error < best_error:
            best_model = model
            best_scaler = scaler
            best_error = error

    best_model.save(f"models/{symbol}.keras")

    temp_model_path = f"{symbol}.keras"
    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)
    
    print(f"Model saved: {symbol} with error: {best_error:.2f}\n")
