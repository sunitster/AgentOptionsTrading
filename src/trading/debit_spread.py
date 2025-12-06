from .base_strategy import BaseStrategy

class DebitSpreadStrategy(BaseStrategy):
    def generate_candidate(self):
        return {
            "type": "debit",
            "direction": None,
            "short_strike": None,
            "long_strike": None,
        }

    def score(self, candidate):
        return 0.0

    def build_orders(self, candidate):
        return []
