from .base_strategy import BaseStrategy

class EventStrangleStrategy(BaseStrategy):
    def generate_candidate(self):
        return {
            "type": "strangle",
            "expiry": None,
            "strike_call": None,
            "strike_put": None,
        }

    def score(self, candidate):
        return 0.0

    def build_orders(self, candidate):
        return []
