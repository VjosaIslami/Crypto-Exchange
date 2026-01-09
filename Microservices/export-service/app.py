from flask import Flask, request, send_file, jsonify
import requests
import csv
import pandas as pd
from datetime import datetime
import io
import json

app = Flask(__name__)

# Service URLs
DATA_SERVICE_URL = "http://localhost:5001"
ANALYSIS_SERVICE_URL = "http://localhost:5002"


@app.route('/api/export/csv', methods=['GET', 'POST'])
def export_csv():
    """Export data as CSV"""
    if request.method == 'GET':
        symbols = request.args.get('symbols', '')
    else:
        data = request.json or {}
        symbols = data.get('symbols', '')

    symbol_list = [s.strip() for s in symbols.split(',') if s.strip()] if symbols else []

    try:
        # Fetch data from data service
        all_data = []

        if symbol_list:
            for symbol in symbol_list:
                response = requests.get(f"{DATA_SERVICE_URL}/api/ohlcv/{symbol}")
                if response.status_code == 200:
                    data = response.json()
                    for item in data:
                        item['symbol'] = symbol
                        all_data.append(item)
        else:
            # Get all symbols first
            response = requests.get(f"{DATA_SERVICE_URL}/api/symbols")
            if response.status_code == 200:
                symbols = response.json()
                for symbol in symbols[:10]:  # Limit for demo
                    response = requests.get(f"{DATA_SERVICE_URL}/api/ohlcv/{symbol}")
                    if response.status_code == 200:
                        data = response.json()
                        for item in data:
                            item['symbol'] = symbol
                            all_data.append(item)

        if not all_data:
            return jsonify({"error": "No data available"}), 400

        # Create CSV
        df = pd.DataFrame(all_data)

        # Create CSV in memory
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        filename = f"crypto_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/export/json', methods=['GET', 'POST'])
def export_json():
    """Export data as JSON"""
    if request.method == 'GET':
        symbols = request.args.get('symbols', '')
    else:
        data = request.json or {}
        symbols = data.get('symbols', '')

    symbol_list = [s.strip() for s in symbols.split(',') if s.strip()] if symbols else []

    try:
        all_data = {}

        if symbol_list:
            for symbol in symbol_list:
                # Get OHLCV data
                response = requests.get(f"{DATA_SERVICE_URL}/api/ohlcv/{symbol}")
                if response.status_code == 200:
                    all_data[symbol] = {
                        'ohlcv': response.json(),
                        'analysis': {}
                    }

                # Get analysis data
                response = requests.get(f"{ANALYSIS_SERVICE_URL}/api/combined/{symbol}")
                if response.status_code == 200:
                    all_data[symbol]['analysis'] = response.json()

        if not all_data:
            return jsonify({"error": "No data available"}), 400

        # Create JSON file
        filename = f"crypto_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        return send_file(
            io.BytesIO(json.dumps(all_data, indent=2).encode('utf-8')),
            mimetype='application/json',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/export/report/<symbol>', methods=['GET'])
def export_report(symbol):
    """Generate a comprehensive report for a symbol"""
    try:
        # Fetch data from both services
        ohlcv_response = requests.get(f"{DATA_SERVICE_URL}/api/ohlcv/{symbol}")
        analysis_response = requests.get(f"{ANALYSIS_SERVICE_URL}/api/combined/{symbol}")

        if ohlcv_response.status_code != 200:
            return jsonify({"error": "No OHLCV data available"}), 400

        if analysis_response.status_code != 200:
            return jsonify({"error": "No analysis data available"}), 400

        ohlcv_data = ohlcv_response.json()
        analysis_data = analysis_response.json()

        # Create HTML report
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Crypto Report - {symbol}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .signal {{ font-weight: bold; padding: 10px; border-radius: 5px; }}
                .buy {{ background: #d4edda; color: #155724; }}
                .sell {{ background: #f8d7da; color: #721c24; }}
                .hold {{ background: #fff3cd; color: #856404; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Crypto Analysis Report</h1>
                <h2>Symbol: {symbol}</h2>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div class="section">
                <h3>Analysis Summary</h3>
                <div class="signal {analysis_data.get('combined_signal', '').lower().replace('_', '-')}">
                    Signal: {analysis_data.get('combined_signal', 'N/A')}
                </div>
            </div>

            <div class="section">
                <h3>Latest OHLCV Data</h3>
        """

        if ohlcv_data:
            latest = ohlcv_data[0]
            html_report += f"""
                <table>
                    <tr><th>Date</th><td>{latest.get('date', 'N/A')}</td></tr>
                    <tr><th>Close</th><td>${latest.get('close', 'N/A'):.2f}</td></tr>
                    <tr><th>High</th><td>${latest.get('high', 'N/A'):.2f}</td></tr>
                    <tr><th>Low</th><td>${latest.get('low', 'N/A'):.2f}</td></tr>
                    <tr><th>Volume</th><td>{latest.get('volume', 'N/A'):.0f}</td></tr>
                </table>
            """

        html_report += """
            </div>

            <div class="section">
                <h3>Technical Indicators</h3>
        """

        if 'technical' in analysis_data:
            tech = analysis_data['technical']
            html_report += f"""
                <table>
                    <tr><th>RSI</th><td>{tech.get('rsi', 'N/A'):.2f}</td></tr>
                    <tr><th>SMA (20)</th><td>${tech.get('sma_20', 'N/A'):.2f}</td></tr>
                    <tr><th>MACD</th><td>{tech.get('macd', 'N/A'):.4f}</td></tr>
                </table>
            """

        html_report += """
            </div>
        </body>
        </html>
        """

        filename = f"{symbol}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

        return send_file(
            io.BytesIO(html_report.encode('utf-8')),
            mimetype='text/html',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "export"})


if __name__ == '__main__':
    app.run(debug=True, port=5003, host='0.0.0.0')