# Base interface for all strategies
class BaseStrategy:
    def __init__(self, symbol, options_chain, context=None):
        self.symbol = symbol
        self.chain = options_chain
        self.ctx = context or {}

    def generate_candidate(self):
        """Return candidate object (spread, strikes, expiry etc.)"""
        raise NotImplementedError

    def score(self, candidate):
        """Return ML score or rule-based score"""
        raise NotImplementedError

    def risk_check(self, candidate, portfolio_state):
        """Return bool allowed or rejected"""
        return True

    def build_orders(self, candidate):
        """Return order legs, qty, price"""
        raise NotImplementedError
