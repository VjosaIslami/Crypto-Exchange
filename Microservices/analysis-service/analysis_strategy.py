from abc import ABC, abstractmethod
from typing import Dict, Any
import os

# Import the analysis modules
from .technical_analysis import compute_technical_analysis
from .lstm_prediction import compute_lstm_analysis
from .sentiment_analysis import compute_onchain_sentiment_analysis


class AnalysisStrategy(ABC):
    """
    Strategy interface for all analysis types.
    Each strategy must implement the analyze method.
    """

    @abstractmethod
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """Analyze a symbol and return results."""
        pass


class TechnicalAnalysisStrategy(AnalysisStrategy):
    """Strategy for technical indicator-based analysis."""

    def analyze(self, symbol: str) -> Dict[str, Any]:
        """Run technical analysis for the given symbol."""
        try:
            return compute_technical_analysis(symbol)
        except Exception as e:
            return {
                "symbol": symbol,
                "error": f"Technical analysis failed: {str(e)}",
                "analysis_type": "technical"
            }


class LstmAnalysisStrategy(AnalysisStrategy):
    """Strategy for LSTM-based price prediction."""

    def analyze(self, symbol: str) -> Dict[str, Any]:
        """Run LSTM prediction for the given symbol."""
        try:
            return compute_lstm_analysis(symbol)
        except Exception as e:
            return {
                "symbol": symbol,
                "error": f"LSTM analysis failed: {str(e)}",
                "analysis_type": "lstm"
            }


class SentimentAnalysisStrategy(AnalysisStrategy):
    """Strategy for on-chain and sentiment analysis."""

    def analyze(self, symbol: str) -> Dict[str, Any]:
        """Run sentiment analysis for the given symbol."""
        try:
            return compute_onchain_sentiment_analysis(symbol)
        except Exception as e:
            return {
                "symbol": symbol,
                "error": f"Sentiment analysis failed: {str(e)}",
                "analysis_type": "sentiment"
            }


class AnalysisContext:
    """
    Context class that uses a strategy pattern to execute analysis.
    Allows runtime strategy switching.
    """

    def __init__(self, strategy: AnalysisStrategy = None):
        self._strategy = strategy

    @property
    def strategy(self) -> AnalysisStrategy:
        """Get the current strategy."""
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: AnalysisStrategy) -> None:
        """Set a new strategy."""
        self._strategy = strategy

    def analyze(self, symbol: str) -> Dict[str, Any]:
        """Execute analysis using the current strategy."""
        if not self._strategy:
            raise ValueError("No analysis strategy has been set")

        if not symbol or not symbol.strip():
            raise ValueError("Symbol cannot be empty")

        return self._strategy.analyze(symbol.strip().upper())


# Factory function to create strategies
def create_strategy(strategy_type: str) -> AnalysisStrategy:
    """Factory function to create analysis strategies."""
    strategy_map = {
        "technical": TechnicalAnalysisStrategy,
        "lstm": LstmAnalysisStrategy,
        "sentiment": SentimentAnalysisStrategy
    }

    strategy_class = strategy_map.get(strategy_type.lower())
    if not strategy_class:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    return strategy_class()


# Utility function to run multiple analyses
def run_multiple_analyses(symbol: str, strategies: list) -> Dict[str, Any]:
    """Run multiple analysis strategies on a symbol."""
    results = {}

    for strategy_type in strategies:
        try:
            strategy = create_strategy(strategy_type)
            context = AnalysisContext(strategy)
            results[strategy_type] = context.analyze(symbol)
        except Exception as e:
            results[strategy_type] = {
                "error": str(e),
                "symbol": symbol
            }

    return {
        "symbol": symbol,
        "analyses": results,
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }