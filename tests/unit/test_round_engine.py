import pandas as pd
import pytest

from core.agent import HouseholdAgent
from core.logger import Logger
from core.round_engine import RoundEngine
from core.schemas import AgentAction


PARAMS = {
    "grid_capacity_kwh": 20.0,
    "unit_utility": 1.0,
    "share_bonus": 0.5,
    "penalty_multiplier": 3.0,
    "num_agents": 3,
    "season_hint": "summer: high solar",
    "total_rounds": 10,
}

_FIXED_ACTION = AgentAction(
    draw_kwh=4.0, offer_kwh=1.0, store_kwh=0.5, reasoning="mock"
)


def _make_row() -> pd.Series:
    return pd.Series(
        {
            "date": "2022-07-01",
            "h1_kwh": 4.0,
            "h6_kwh": 1.5,
            "h8_kwh": 5.0,
            "solar_w_m2": 600.0,
            "h1_pv_kwh": 10.8,
            "h6_pv_kwh": 6.48,
            "h8_pv_kwh": 17.28,
        }
    )


def _make_engine(mocker, tmp_path) -> RoundEngine:
    mock_llm = mocker.MagicMock()
    mock_llm.generate.return_value = _FIXED_ACTION

    agents = [
        HouseholdAgent("H1", 25.0, 10.0, "persona A", mock_llm),
        HouseholdAgent("H6", 15.0, 6.0, "persona B", mock_llm),
        HouseholdAgent("H8", 40.0, 15.0, "persona C", mock_llm),
    ]
    logger = Logger(tmp_path / "run")
    return RoundEngine(agents, PARAMS, logger)


def test_run_round_returns_result(mocker, tmp_path):
    engine = _make_engine(mocker, tmp_path)
    result = engine.run_round(1, _make_row())

    assert result.round_num == 1
    assert result.total_draw_kwh == pytest.approx(4.0 * 3)  # 3 agents × 4.0
    assert len(result.payoffs) == 3


def test_no_overflow_when_under_capacity(mocker, tmp_path):
    engine = _make_engine(mocker, tmp_path)
    result = engine.run_round(1, _make_row())

    # 3 agents × 4.0 = 12 kWh < 20 kWh capacity
    assert result.overflow_kwh == 0.0
    for p in result.payoffs:
        assert p.penalty == 0.0


def test_memory_updated_after_round(mocker, tmp_path):
    engine = _make_engine(mocker, tmp_path)
    engine.run_round(1, _make_row())

    for agent in engine._agents:
        assert len(agent.memory._own_history) == 1


def test_agent_states_accumulate(mocker, tmp_path):
    engine = _make_engine(mocker, tmp_path)
    engine.run_round(1, _make_row())
    engine.run_round(2, _make_row())

    for state in engine.agent_states:
        assert state.rounds_played == 2
        assert state.total_net_payoff > 0


def test_prev_actions_passed_as_neighbor_context(mocker, tmp_path):
    engine = _make_engine(mocker, tmp_path)
    engine.run_round(1, _make_row())

    # After round 1, prev_actions should be populated
    for aid, action in engine._prev_actions.items():
        assert action is not None
        assert action.draw_kwh == 4.0
