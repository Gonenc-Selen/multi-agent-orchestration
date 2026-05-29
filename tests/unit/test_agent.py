import pytest

from core.agent import HouseholdAgent
from core.schemas import AgentAction, RoundContext


SCENARIO_PARAMS = {
    "num_agents": 3,
    "grid_capacity_kwh": 20.0,
    "unit_utility": 1.0,
    "share_bonus": 0.5,
    "penalty_multiplier": 3.0,
    "total_rounds": 10,
}


def _make_context(round_num: int = 1) -> RoundContext:
    return RoundContext(
        round_num=round_num,
        agent_id="H1",
        consumption_kwh=4.0,
        pv_kwh=10.0,
        battery_soc=0.5,
        battery_capacity_kwh=10.0,
        neighbor_actions={"H6": None, "H8": None},
        memory_summary="No history yet.",
        season_hint="summer: high solar",
    )


def _make_agent(mock_llm) -> HouseholdAgent:
    return HouseholdAgent(
        agent_id="H1",
        panel_area_m2=25.0,
        battery_capacity_kwh=10.0,
        persona="Large family, heavy AC use.",
        llm_client=mock_llm,
    )


def test_decide_returns_agent_action(mocker):
    mock_llm = mocker.MagicMock()
    mock_llm.generate.return_value = AgentAction(
        draw_kwh=4.0, offer_kwh=2.0, store_kwh=1.0, reasoning="Planned response."
    )
    agent = _make_agent(mock_llm)
    action = agent.decide(_make_context(), SCENARIO_PARAMS)

    assert isinstance(action, AgentAction)
    assert action.draw_kwh == 4.0
    mock_llm.generate.assert_called_once()


def test_decide_falls_back_on_llm_failure(mocker):
    mock_llm = mocker.MagicMock()
    mock_llm.generate.side_effect = ValueError("API error")
    agent = _make_agent(mock_llm)
    action = agent.decide(_make_context(), SCENARIO_PARAMS)

    # Default action: draw = consumption_kwh, offer = 0, store = 0
    assert action.draw_kwh == 4.0
    assert action.offer_kwh == 0.0
    assert action.store_kwh == 0.0
    assert "Default" in action.reasoning


def test_battery_charge_clips_at_capacity():
    mock_llm = object()  # not called
    agent = HouseholdAgent("H1", 25.0, 10.0, "test", mock_llm)  # type: ignore
    agent.battery_soc = 0.9  # 90% full → 1 kWh space left

    actual = agent.update_battery_charge(5.0)  # try to store 5, only 1 fits
    assert actual < 5.0
    assert agent.battery_soc <= 1.0


def test_battery_discharge_does_not_go_negative():
    mock_llm = object()
    agent = HouseholdAgent("H1", 25.0, 10.0, "test", mock_llm)  # type: ignore
    agent.battery_soc = 0.1  # nearly empty

    agent.discharge_battery(100.0)  # ask for way more than available
    assert agent.battery_soc >= 0.0


def test_battery_soc_starts_at_50_percent():
    mock_llm = object()
    agent = HouseholdAgent("H1", 25.0, 10.0, "test", mock_llm)  # type: ignore
    assert agent.battery_soc == 0.5
