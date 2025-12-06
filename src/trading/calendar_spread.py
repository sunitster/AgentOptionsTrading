from .base_strategy import BaseStrategy

class CalendarSpreadStrategy(BaseStrategy):
    def generate_candidate(self):
        # prototype structure, no trading logic yet
        return {
            "type": "calendar",
            "strike": None,
            "near_expiry": None,
            "far_expiry": None,
        }

    def score(self, candidate):
        # default placeholder
        return 0.0

    def build_orders(self, candidate):
        return []  # blank for now
