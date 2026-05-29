import pandas as pd
import pytest

from core.schemas import AgentAction


@pytest.fixture
def default_action() -> AgentAction:
    return AgentAction(draw_kwh=3.0, offer_kwh=1.0, store_kwh=0.5, reasoning="test")


@pytest.fixture
def sample_community_df() -> pd.DataFrame:
    """Minimal in-memory community_window.csv for testing."""
    return pd.DataFrame(
        {
            "date": ["2022-07-01", "2022-07-02"],
            "h1_kwh": [4.0, 3.5],
            "h6_kwh": [1.5, 1.8],
            "h8_kwh": [5.0, 4.8],
            "solar_w_m2": [600.0, 550.0],
            "h1_pv_kwh": [10.8, 9.9],
            "h6_pv_kwh": [6.48, 5.94],
            "h8_pv_kwh": [17.28, 15.84],
        }
    )
