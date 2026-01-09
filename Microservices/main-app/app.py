from datetime import datetime

from flask import Flask, render_template, request, jsonify
import requests
import os

app = Flask(__name__)

# Microservice URLs
DATA_SERVICE_URL = os.getenv('DATA_SERVICE_URL', 'http://localhost:5001')
ANALYSIS_SERVICE_URL = os.getenv('ANALYSIS_SERVICE_URL', 'http://localhost:5002')
EXPORT_SERVICE_URL = os.getenv('EXPORT_SERVICE_URL', 'http://localhost:5003')


@app.route('/')
def dashboard():
    return render_template('dashboard.html')


@app.route('/api/market-data')
def api_market_data():
    """Aggregate market data from data service"""
    try:
        response = requests.get(f"{DATA_SERVICE_URL}/api/stats")
        if response.status_code == 200:
            return jsonify(response.json())
    except Exception as e:
        print(f"Error fetching market data: {e}")

    return jsonify({"error": "Service unavailable"}), 503


@app.route('/api/symbols')
def api_symbols():
    """Get symbols from data service"""
    try:
        response = requests.get(f"{DATA_SERVICE_URL}/api/symbols")
        if response.status_code == 200:
            return jsonify(response.json())
    except Exception as e:
        print(f"Error fetching symbols: {e}")

    return jsonify([])


@app.route('/api/technical-analysis')
def api_technical_analysis():
    """Forward to analysis service"""
    symbol = request.args.get('symbol', '').strip().upper()
    if not symbol:
        return jsonify({"error": "Missing symbol parameter"}), 400

    try:
        response = requests.get(f"{ANALYSIS_SERVICE_URL}/api/technical/{symbol}")
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/sentiment-analysis')
def api_sentiment_analysis():
    """Forward to analysis service"""
    symbol = request.args.get('symbol', '').strip().upper()
    if not symbol:
        return jsonify({"error": "Missing symbol parameter"}), 400

    try:
        response = requests.get(f"{ANALYSIS_SERVICE_URL}/api/sentiment/{symbol}")
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/combined-analysis')
def api_combined_analysis():
    """Forward to analysis service"""
    symbol = request.args.get('symbol', '').strip().upper()
    if not symbol:
        return jsonify({"error": "Missing symbol parameter"}), 400

    try:
        response = requests.get(f"{ANALYSIS_SERVICE_URL}/api/combined/{symbol}")
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/export/csv')
def api_export_csv():
    """Forward to export service"""
    symbols = request.args.get('symbols', '')

    try:
        response = requests.get(f"{EXPORT_SERVICE_URL}/api/export/csv",
                                params={'symbols': symbols})

        if response.status_code == 200:
            # Return the file download
            return response.content, 200, {
                'Content-Type': 'text/csv',
                'Content-Disposition': response.headers.get('Content-Disposition', 'attachment')
            }
        else:
            return jsonify(response.json()), response.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/export/report/<symbol>')
def api_export_report(symbol):
    """Forward to export service"""
    try:
        response = requests.get(f"{EXPORT_SERVICE_URL}/api/export/report/{symbol}")

        if response.status_code == 200:
            return response.content, 200, {
                'Content-Type': 'text/html',
                'Content-Disposition': response.headers.get('Content-Disposition', 'attachment')
            }
        else:
            return jsonify(response.json()), response.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/update', methods=['POST'])
def trigger_update():
    """Trigger data update in data service"""
    try:
        symbols = request.json.get('symbols', []) if request.json else []
        response = requests.post(f"{DATA_SERVICE_URL}/api/update",
                                 json={'symbols': symbols})
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/health')
def health_check():
    """Check health of all microservices"""
    services = {
        'data_service': DATA_SERVICE_URL,
        'analysis_service': ANALYSIS_SERVICE_URL,
        'export_service': EXPORT_SERVICE_URL
    }

    health_status = {}

    for name, url in services.items():
        try:
            response = requests.get(f"{url}/health", timeout=5)
            health_status[name] = {
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'response_time': response.elapsed.total_seconds()
            }
        except Exception as e:
            health_status[name] = {
                'status': 'unreachable',
                'error': str(e)
            }

    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'services': health_status
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')