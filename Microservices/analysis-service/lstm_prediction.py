import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import requests
import warnings
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# Data service URL
DATA_SERVICE_URL = os.getenv('DATA_SERVICE_URL', 'http://localhost:5001')


def load_price_data(symbol, days=365):
    """Load price data from data service"""
    try:
        response = requests.get(f"{DATA_SERVICE_URL}/api/ohlcv/{symbol}", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df.sort_index()
                if len(df) > days:
                    df = df.iloc[-days:]
                return df
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading price data for LSTM ({symbol}): {e}")
        return pd.DataFrame()


def _create_sequences(features_scaled, target_scaled, lookback):
    X, y = [], []
    for i in range(len(features_scaled) - lookback):
        X.append(features_scaled[i:i + lookback])
        y.append(target_scaled[i + lookback])

    if len(X) == 0:
        return np.array([]), np.array([])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y


def _compute_mape(y_true, y_pred):
    """Compute Mean Absolute Percentage Error"""
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    # Avoid division by zero
    mask = y_true != 0
    if not np.any(mask):
        return 100.0

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def create_lstm_model(input_shape, units=50, dropout_rate=0.2):
    """Create LSTM model architecture"""
    model = Sequential([
        LSTM(units, input_shape=input_shape, return_sequences=False),
        Dropout(dropout_rate),
        Dense(32, activation="relu"),
        Dropout(dropout_rate),
        Dense(16, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


def train_lstm_for_symbol(
        symbol: str,
        days: int = 365,
        lookback: int = 30,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
):
    """Train LSTM model for a specific symbol"""
    # Load historical data
    df = load_price_data(symbol, days=days)

    if df.empty or len(df) < 100:
        return {
            "symbol": symbol,
            "error": f"Not enough OHLCV data to train LSTM. Only {len(df)} records available."
        }

    # Check required columns
    required_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return {
            "symbol": symbol,
            "error": f"Missing required columns: {missing_cols}"
        }

    # Prepare features and target
    features = df[required_cols].values.astype(np.float32)
    close_prices = df["close"].values.reshape(-1, 1).astype(np.float32)

    # Handle NaN values
    if np.isnan(features).any() or np.isnan(close_prices).any():
        # Simple forward fill for missing values
        features = pd.DataFrame(features).fillna(method='ffill').fillna(method='bfill').values
        close_prices = pd.DataFrame(close_prices).fillna(method='ffill').fillna(method='bfill').values

    # Scale features and target
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    features_scaled = scaler_X.fit_transform(features).astype(np.float32)
    close_scaled = scaler_y.fit_transform(close_prices).astype(np.float32).flatten()

    # Build sequences for time-series
    X, y = _create_sequences(features_scaled, close_scaled, lookback)

    if len(X) < 20:
        return {
            "symbol": symbol,
            "error": f"Not enough sequence samples after lookback for LSTM. Need at least 20, got {len(X)}."
        }

    # Train/test split
    split_idx = int(len(X) * (1 - validation_split))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if len(X_test) == 0 or len(X_train) == 0:
        return {
            "symbol": symbol,
            "error": "Train or test set is empty; not enough data."
        }

    n_features = X.shape[2]

    # Create and train model
    model = create_lstm_model((lookback, n_features))

    # Train with early stopping callback (simulated)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        shuffle=False
    )

    # Evaluate on test set
    y_test_pred_scaled = model.predict(X_test, verbose=0)

    # Inverse-transform to real prices
    y_test_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled).flatten()

    # Calculate metrics
    mse = mean_squared_error(y_test_true, y_test_pred)
    rmse = float(np.sqrt(mse))
    mape = _compute_mape(y_test_true, y_test_pred)
    r2 = float(r2_score(y_test_true, y_test_pred))

    # Calculate direction accuracy
    direction_true = np.diff(y_test_true) > 0
    direction_pred = np.diff(y_test_pred) > 0
    if len(direction_true) > 0:
        direction_accuracy = float(np.mean(direction_true == direction_pred)) * 100
    else:
        direction_accuracy = 0.0

    # Predict next price
    last_seq = features_scaled[-lookback:]
    last_seq = last_seq.reshape(1, lookback, n_features).astype(np.float32)

    next_price_scaled = model.predict(last_seq, verbose=0)[0][0]
    next_price = scaler_y.inverse_transform(np.array([[next_price_scaled]])).flatten()[0]

    # Calculate confidence based on recent prediction accuracy
    recent_true = y_test_true[-min(10, len(y_test_true)):]
    recent_pred = y_test_pred[-min(10, len(y_test_pred)):]
    recent_mape = _compute_mape(recent_true, recent_pred)
    confidence = max(0, 100 - recent_mape) / 100  # Normalize to 0-1

    # Results in JSON
    return {
        "symbol": symbol,
        "model_info": {
            "lookback": lookback,
            "features": n_features,
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "epochs_trained": epochs
        },
        "metrics": {
            "mse": float(mse),
            "rmse": rmse,
            "mape": mape,
            "r2": r2,
            "direction_accuracy": direction_accuracy
        },
        "forecast": {
            "next_day_close": float(next_price),
            "current_close": float(close_prices[-1][0]),
            "predicted_change_percent": float((next_price - close_prices[-1][0]) / close_prices[-1][0] * 100),
            "confidence": float(confidence)
        },
        "recent_performance": {
            "last_true_price": float(y_test_true[-1]),
            "last_pred_price": float(y_test_pred[-1]),
            "last_true_date": df.index[-1].strftime('%Y-%m-%d')
        },
        "source_service": "analysis-service",
        "analysis_type": "lstm_prediction",
        "timestamp": datetime.now().isoformat()
    }


def compute_lstm_analysis(symbol: str):
    """Public interface for LSTM analysis"""
    try:
        return train_lstm_for_symbol(symbol)
    except Exception as e:
        print(f"LSTM analysis failed for {symbol}: {e}")
        return {
            "symbol": symbol,
            "error": f"LSTM analysis failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }