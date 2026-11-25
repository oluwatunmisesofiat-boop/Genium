"""SCD Type 2 ProductVersion lifecycle and transition window management."""
from datetime import date
from typing import Optional


class VersionLifecycleManager:
    @staticmethod
    def is_within_transition_window(valid_to: Optional[date], shelf_life_months: int = 24) -> bool:
        """Determines if a superseded version is still eligible for genuine match."""
        if valid_to is None:
            return True  # Active version
        today = date.today()
        # Days approximation: 30 days per month
        max_transition_days = shelf_life_months * 30
        return (today - valid_to).days <= max_transition_days
