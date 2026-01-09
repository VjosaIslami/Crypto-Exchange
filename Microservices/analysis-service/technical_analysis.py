import numpy as np
import pandas as pd
import requests
from datetime import datetime
import os

# Data service URL
DATA_SERVICE_URL = os.getenv('DATA_SERVICE_URL', 'http://localhost:5001')


def load_price_data(symbol, days=365):
    """Load price data from data service instead of local database"""
    try:
        response = requests.get(f"{DATA_SERVICE_URL}/api/ohlcv/{symbol}", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df.sort_index()
                if len(df) > days:
                    df = df.iloc[-days:]
                return df
            else:
                print(f"No data returned for symbol: {symbol}")
        else:
            print(f"Failed to fetch data for {symbol}: HTTP {response.status_code}")
    except requests.exceptions.Timeout:
        print(f"Timeout fetching data for {symbol}")
    except requests.exceptions.ConnectionError:
        print(f"Connection error fetching data for {symbol}")
    except Exception as e:
        print(f"Error loading price data for {symbol}: {e}")
    return pd.DataFrame()


def resample_ohlcv(df, timeframe):
    if df.empty:
        return df

    rule_map = {
        '1D': 'D',
        '1W': 'W',
        '1M': 'M',
    }
    rule = rule_map.get(timeframe)
    if rule is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    # Make sure required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    ohlcv = df[required_cols]

    resampled = ohlcv.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna()

    return resampled


def add_sma(df, window=20):
    if 'close' not in df.columns:
        return df
    df[f'sma_{window}'] = df['close'].rolling(window, min_periods=1).mean()
    return df


def add_ema(df, window=20):
    if 'close' not in df.columns:
        return df
    df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False, min_periods=1).mean()
    return df


def add_wma(df, window=20):
    if 'close' not in df.columns:
        return df
    weights = np.arange(1, window + 1)

    def wma_func(prices):
        if len(prices) < window:
            return np.nan
        return np.dot(prices, weights) / weights.sum()

    df[f'wma_{window}'] = df['close'].rolling(window).apply(wma_func, raw=True)
    return df


def add_bollinger(df, window=20, num_std=2):
    if 'close' not in df.columns:
        return df
    sma = df['close'].rolling(window, min_periods=1).mean()
    std = df['close'].rolling(window, min_periods=1).std()
    df[f'bb_middle_{window}'] = sma
    df[f'bb_upper_{window}'] = sma + num_std * std
    df[f'bb_lower_{window}'] = sma - num_std * std
    return df


def add_volume_sma(df, window=20):
    if 'volume' not in df.columns:
        return df
    df[f'vol_sma_{window}'] = df['volume'].rolling(window, min_periods=1).mean()
    return df


def add_rsi(df, window=14):
    if 'close' not in df.columns:
        return df
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window, min_periods=1).mean()
    avg_loss = loss.rolling(window, min_periods=1).mean()

    # Avoid division by zero
    avg_loss = avg_loss.replace(0, np.nan)
    rs = avg_gain / avg_loss

    df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
    df[f'rsi_{window}'] = df[f'rsi_{window}'].clip(0, 100)  # Clip to 0-100 range
    return df


def add_macd(df, fast=12, slow=26, signal=9):
    if 'close' not in df.columns:
        return df
    ema_fast = df['close'].ewm(span=fast, adjust=False, min_periods=1).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False, min_periods=1).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False, min_periods=1).mean()
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd - macd_signal
    return df


def add_stochastic(df, window=14):
    if not all(col in df.columns for col in ['high', 'low', 'close']):
        return df
    lowest_low = df['low'].rolling(window, min_periods=1).min()
    highest_high = df['high'].rolling(window, min_periods=1).max()

    # Avoid division by zero
    denominator = highest_high - lowest_low
    denominator = denominator.replace(0, np.nan)

    df['stoch_k'] = 100 * (df['close'] - lowest_low) / denominator
    df['stoch_k'] = df['stoch_k'].clip(0, 100)  # Clip to 0-100 range
    df['stoch_d'] = df['stoch_k'].rolling(3, min_periods=1).mean()
    return df


def add_cci(df, window=20):
    if not all(col in df.columns for col in ['high', 'low', 'close']):
        return df
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = tp.rolling(window, min_periods=1).mean()
    mad = (tp - sma_tp).abs().rolling(window, min_periods=1).mean()

    # Avoid division by zero
    mad = mad.replace(0, np.nan)
    df['cci'] = (tp - sma_tp) / (0.015 * mad)
    return df


def add_adx(df, window=14):
    if not all(col in df.columns for col in ['high', 'low', 'close']):
        return df

    high = df['high']
    low = df['low']
    close = df['close']

    # Calculate +DM and -DM
    plus_dm = high.diff()
    minus_dm = low.diff() * -1

    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

    # Calculate True Range
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window, min_periods=1).sum()

    # Calculate +DI and -DI
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(window, min_periods=1).sum() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(window, min_periods=1).sum() / atr)

    # Calculate DX and ADX
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.rolling(window, min_periods=1).mean()

    df['adx'] = adx
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di

    return df


def add_all_indicators(df):
    """Add all technical indicators to dataframe"""
    if df.empty:
        return df

    df = add_sma(df, 20)
    df = add_ema(df, 20)
    df = add_wma(df, 20)
    df = add_bollinger(df, 20)
    df = add_volume_sma(df, 20)
    df = add_rsi(df, 14)
    df = add_macd(df)
    df = add_stochastic(df)
    df = add_cci(df)
    df = add_adx(df, 14)

    return df


def generate_signal_row(row):
    """Generate trading signal based on indicator values"""
    price = row['close']
    sma20 = row.get('sma_20', np.nan)
    rsi14 = row.get('rsi_14', np.nan)
    macd_hist = row.get('macd_hist', np.nan)
    stoch_k = row.get('stoch_k', np.nan)
    adx = row.get('adx', np.nan)

    signal = 'HOLD'
    reason = []

    # Buy signals
    buy_signals = 0
    if not np.isnan(rsi14) and rsi14 < 30:
        reason.append('RSI < 30 (oversold)')
        buy_signals += 1
    if not np.isnan(macd_hist) and macd_hist > 0:
        reason.append('MACD histogram > 0 (bullish momentum)')
        buy_signals += 1
    if not np.isnan(stoch_k) and stoch_k < 20:
        reason.append('Stochastic < 20 (oversold)')
        buy_signals += 1
    if not np.isnan(adx) and adx > 20:
        reason.append('ADX > 20 (trend strength)')
        buy_signals += 0.5
    if not np.isnan(sma20) and price > sma20:
        reason.append('Price above SMA20 (uptrend)')
        buy_signals += 1

    # Sell signals
    sell_signals = 0
    sell_reason = []
    if not np.isnan(rsi14) and rsi14 > 70:
        sell_reason.append('RSI > 70 (overbought)')
        sell_signals += 1
    if not np.isnan(macd_hist) and macd_hist < 0:
        sell_reason.append('MACD histogram < 0 (bearish momentum)')
        sell_signals += 1
    if not np.isnan(stoch_k) and stoch_k > 80:
        sell_reason.append('Stochastic > 80 (overbought)')
        sell_signals += 1
    if not np.isnan(adx) and adx > 20:
        sell_reason.append('ADX > 20 (trend strength)')
        sell_signals += 0.5
    if not np.isnan(sma20) and price < sma20:
        sell_reason.append('Price below SMA20 (downtrend)')
        sell_signals += 1

    # Determine final signal
    if buy_signals >= 2 and buy_signals > sell_signals:
        signal = 'BUY'
        reason = reason
    elif sell_signals >= 2 and sell_signals > buy_signals:
        signal = 'SELL'
        reason = sell_reason
    else:
        signal = 'HOLD'
        reason = ['Mixed or weak signals']

    return pd.Series({'signal': signal, 'signal_reason': '; '.join(reason)})


def add_signals(df):
    """Add trading signals to dataframe"""
    if df.empty:
        return df

    signals = df.apply(generate_signal_row, axis=1)
    df['signal'] = signals['signal']
    df['signal_reason'] = signals['signal_reason']
    return df


def compute_technical_analysis(symbol, days=365, timeframes=('1D', '1W', '1M')):
    """Compute technical analysis for a symbol"""
    base_df = load_price_data(symbol, days=days)
    if base_df.empty:
        return {
            'symbol': symbol,
            'timeframes': {},
            'error': f'No OHLCV data available from data service for symbol: {symbol}',
            'timestamp': datetime.now().isoformat()
        }

    results = {}

    for tf in timeframes:
        try:
            tf_df = resample_ohlcv(base_df, tf)
            if tf_df.empty:
                continue

            tf_df = add_all_indicators(tf_df)
            tf_df = add_signals(tf_df)

            last_row = tf_df.iloc[-1].to_dict()

            # Prepare indicator values
            indicators = {}
            indicator_keys = [
                'rsi_14', 'macd', 'macd_hist', 'stoch_k', 'adx', 'cci',
                'sma_20', 'ema_20', 'wma_20', 'bb_upper_20', 'bb_lower_20', 'vol_sma_20'
            ]

            for key in indicator_keys:
                value = last_row.get(key, np.nan)
                if pd.notna(value):
                    indicators[key] = float(value)
                else:
                    indicators[key] = None

            results[tf] = {
                'last_date': tf_df.index[-1].strftime('%Y-%m-%d'),
                'close': float(last_row['close']),
                'open': float(last_row.get('open', 0)),
                'high': float(last_row.get('high', 0)),
                'low': float(last_row.get('low', 0)),
                'volume': float(last_row.get('volume', 0)),
                'indicators': indicators,
                'signal': last_row.get('signal', 'HOLD'),
                'signal_reason': last_row.get('signal_reason', 'No signal generated')
            }
        except Exception as e:
            print(f"Error processing timeframe {tf} for {symbol}: {e}")
            results[tf] = {
                'error': f"Failed to analyze timeframe {tf}: {str(e)}"
            }

    return {
        'symbol': symbol,
        'timeframes': results,
        'source_service': 'analysis-service',
        'analysis_type': 'technical',
        'timestamp': datetime.now().isoformat()
    }