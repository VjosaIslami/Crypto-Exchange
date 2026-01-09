from flask import Flask, request, jsonify
from analysis_strategy import (
    AnalysisContext,
    TechnicalAnalysisStrategy,
    LstmAnalysisStrategy,
    SentimentAnalysisStrategy
)
from datetime import datetime
import os

app = Flask(__name__)

# Configuration
DATA_SERVICE_URL = os.getenv('DATA_SERVICE_URL', 'http://localhost:5001')


@app.route("/api/technical/<symbol>", methods=["GET"])
def api_technical_analysis(symbol):
    """Technical analysis endpoint"""
    if not symbol or not symbol.strip():
        return jsonify({"error": "Missing or invalid 'symbol' parameter"}), 400

    symbol = symbol.strip().upper()

    try:
        context = AnalysisContext(TechnicalAnalysisStrategy())
        result = context.analyze(symbol)
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Technical analysis error for {symbol}: {str(e)}")
        return jsonify({"error": f"Technical analysis failed: {str(e)}", "symbol": symbol}), 500


@app.route("/api/lstm/<symbol>", methods=["GET"])
def api_lstm_prediction(symbol):
    """LSTM prediction endpoint"""
    if not symbol or not symbol.strip():
        return jsonify({"error": "Missing or invalid 'symbol' parameter"}), 400

    symbol = symbol.strip().upper()

    try:
        context = AnalysisContext(LstmAnalysisStrategy())
        result = context.analyze(symbol)
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"LSTM prediction error for {symbol}: {str(e)}")
        return jsonify({"error": f"LSTM prediction failed: {str(e)}", "symbol": symbol}), 500


@app.route("/api/sentiment/<symbol>", methods=["GET"])
def api_sentiment_analysis(symbol):
    """Sentiment analysis endpoint"""
    if not symbol or not symbol.strip():
        return jsonify({"error": "Missing or invalid 'symbol' parameter"}), 400

    symbol = symbol.strip().upper()

    try:
        context = AnalysisContext(SentimentAnalysisStrategy())
        result = context.analyze(symbol)
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Sentiment analysis error for {symbol}: {str(e)}")
        return jsonify({"error": f"Sentiment analysis failed: {str(e)}", "symbol": symbol}), 500


@app.route("/api/combined/<symbol>", methods=["GET"])
def api_combined_analysis(symbol):
    """Get all analysis types combined"""
    if not symbol or not symbol.strip():
        return jsonify({"error": "Missing or invalid 'symbol' parameter"}), 400

    symbol = symbol.strip().upper()

    try:
        # Run all analyses in parallel (in production, could use async)
        strategies = {
            "technical": TechnicalAnalysisStrategy(),
            "lstm": LstmAnalysisStrategy(),
            "sentiment": SentimentAnalysisStrategy()
        }

        results = {}
        for name, strategy in strategies.items():
            try:
                context = AnalysisContext(strategy)
                results[name] = context.analyze(symbol)
            except Exception as e:
                results[name] = {"error": str(e), "symbol": symbol}

        # Determine overall recommendation
        overall_signal = determine_overall_signal(results)

        return jsonify({
            "symbol": symbol,
            "analyses": results,
            "overall_signal": overall_signal,
            "timestamp": datetime.now().isoformat(),
            "source_service": "analysis-service"
        })
    except Exception as e:
        app.logger.error(f"Combined analysis error for {symbol}: {str(e)}")
        return jsonify({"error": f"Combined analysis failed: {str(e)}", "symbol": symbol}), 500


def determine_overall_signal(results):
    """Determine overall trading signal from all analyses"""
    signals = []

    # Check technical analysis signal
    if 'technical' in results and 'error' not in results['technical']:
        if 'timeframes' in results['technical'] and '1D' in results['technical']['timeframes']:
            signal = results['technical']['timeframes']['1D'].get('signal', 'HOLD')
            signals.append(signal)

    # Check sentiment analysis signal
    if 'sentiment' in results and 'error' not in results['sentiment']:
        signal = results['sentiment'].get('final_signal', 'HOLD')
        signals.append(signal)

    # Count signals
    buy_count = sum(1 for s in signals if 'BUY' in s)
    sell_count = sum(1 for s in signals if 'SELL' in s)

    if buy_count >= 2:
        return 'STRONG_BUY'
    elif sell_count >= 2:
        return 'STRONG_SELL'
    elif buy_count > sell_count:
        return 'BUY'
    elif sell_count > buy_count:
        return 'SELL'
    else:
        return 'HOLD'


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    try:
        # Test that we can import and instantiate all strategies
        strategies = [
            TechnicalAnalysisStrategy(),
            LstmAnalysisStrategy(),
            SentimentAnalysisStrategy()
        ]

        return jsonify({
            "status": "healthy",
            "service": "analysis-service",
            "timestamp": datetime.now().isoformat(),
            "strategies": [s.__class__.__name__ for s in strategies],
            "data_service_url": DATA_SERVICE_URL
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "service": "analysis-service",
            "error": str(e)
        }), 500


@app.route("/", methods=["GET"])
def index():
    """Service information page"""
    return jsonify({
        "service": "Crypto Analysis Microservice",
        "version": "1.0.0",
        "endpoints": {
            "/api/technical/<symbol>": "Technical analysis",
            "/api/lstm/<symbol>": "LSTM price prediction",
            "/api/sentiment/<symbol>": "Sentiment analysis",
            "/api/combined/<symbol>": "All analyses combined",
            "/health": "Health check"
        },
        "documentation": "See README for detailed API documentation"
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5002))
    app.run(debug=True, port=port, host="0.0.0.0")