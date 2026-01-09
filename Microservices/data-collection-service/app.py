from flask import Flask, request, jsonify
import requests
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import time
import os
import threading

app = Flask(__name__)

# Database setup
DB_PATH = 'crypto_data.db'


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            UNIQUE(symbol, date)
        )
    """)
    conn.commit()
    conn.close()


# Initialize database on startup
init_database()


def fetch_symbols():
    """Fetch trading symbols from Binance"""
    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        response = requests.get(url)
        data = response.json()
        symbols = []
        for symbol_info in data['symbols']:
            if (symbol_info['status'] == 'TRADING' and
                    symbol_info['quoteAsset'] in ['USDT', 'BUSD', 'BTC', 'ETH'] and
                    not symbol_info['symbol'].endswith('UPUSDT') and
                    not symbol_info['symbol'].endswith('DOWNUSDT')):
                symbols.append(symbol_info['symbol'])
        return symbols[:100]  # Limit for demo
    except Exception as e:
        print(f"Error fetching symbols: {e}")
        return []


def update_symbol_data(symbol):
    """Update data for a specific symbol"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get last date
    cursor.execute("SELECT MAX(date) as last_date FROM ohlcv_data WHERE symbol = ?", (symbol,))
    result = cursor.fetchone()
    last_date = result['last_date']

    if last_date:
        start_date = datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)
    else:
        start_date = datetime.now() - timedelta(days=30)

    # Fetch data from Binance
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': '1d',
        'startTime': int(start_date.timestamp() * 1000),
        'endTime': int(datetime.now().timestamp() * 1000),
        'limit': 1000
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        for item in data:
            date = datetime.fromtimestamp(item[0] / 1000).strftime('%Y-%m-%d')
            cursor.execute("""
                INSERT OR IGNORE INTO ohlcv_data 
                (symbol, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, date,
                float(item[1]), float(item[2]), float(item[3]),
                float(item[4]), float(item[5])
            ))

        conn.commit()
        print(f"Updated {symbol}: {len(data)} records")

    except Exception as e:
        print(f"Error updating {symbol}: {e}")
    finally:
        conn.close()


@app.route('/api/update', methods=['POST'])
def update_data():
    """Trigger data update"""
    data = request.json or {}
    symbols = data.get('symbols', [])

    if not symbols:
        symbols = fetch_symbols()

    # Update in background thread
    def update_background():
        for symbol in symbols:
            update_symbol_data(symbol)
            time.sleep(0.1)  # Rate limiting

    thread = threading.Thread(target=update_background)
    thread.start()

    return jsonify({
        "status": "started",
        "symbols": len(symbols),
        "message": f"Updating {len(symbols)} symbols in background"
    })


@app.route('/api/symbols', methods=['GET'])
def get_symbols():
    """Get list of available symbols"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT symbol FROM ohlcv_data ORDER BY symbol")
    symbols = [row['symbol'] for row in cursor.fetchall()]
    conn.close()
    return jsonify(symbols)


@app.route('/api/ohlcv/<symbol>', methods=['GET'])
def get_ohlcv(symbol):
    """Get OHLCV data for a symbol"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT date, open, high, low, close, volume 
        FROM ohlcv_data 
        WHERE symbol = ? 
        ORDER BY date DESC 
        LIMIT 100
    """, (symbol,))

    data = []
    for row in cursor.fetchall():
        data.append({
            'date': row['date'],
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row['volume']
        })

    conn.close()
    return jsonify(data)


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get database statistics"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(DISTINCT symbol) as total_symbols FROM ohlcv_data")
    total_symbols = cursor.fetchone()['total_symbols']

    cursor.execute("SELECT COUNT(*) as total_records FROM ohlcv_data")
    total_records = cursor.fetchone()['total_records']

    cursor.execute("SELECT MAX(date) as last_updated FROM ohlcv_data")
    last_updated = cursor.fetchone()['last_updated']

    db_size = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0

    conn.close()

    return jsonify({
        'total_symbols': total_symbols,
        'total_records': total_records,
        'last_updated': last_updated,
        'db_size': f"{db_size / (1024 * 1024):.2f} MB"
    })


if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')