"""End-to-end smoke test: 3 agents × 2 rounds with real Vertex AI API.

Run manually:
    uv run pytest tests/integration/ --integration -v

Skipped automatically when --integration flag is absent or
when GOOGLE_CLOUD_PROJECT is empty.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from core.agent import HouseholdAgent
from core.llm_client import LLMClient
from core.logger import Logger
from core.metrics import compute_run_metrics
from core.round_engine import RoundEngine
from core.schemas import AgentAction, RoundResult

# ---------------------------------------------------------------------------
# pytest marker & skip logic
# ---------------------------------------------------------------------------


def _skip_if_no_credentials() -> None:
    """Skip if ADC credentials or project are unavailable."""
    try:
        from core.config import settings

        if not settings.google_cloud_project:
            pytest.skip("GOOGLE_CLOUD_PROJECT not set")
        import google.auth

        google.auth.default()
    except Exception as exc:
        pytest.skip(f"Credentials unavailable: {exc}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SCENARIO_PATH = Path("scenarios/puebla_yaz.yaml")
_DATA_PATH = Path("data/processed/community_window.csv")
_SMOKE_ROUNDS = 2


@pytest.fixture(scope="module")
def scenario() -> dict:  # type: ignore[type-arg]
    with open(_SCENARIO_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def smoke_df(scenario: dict) -> pd.DataFrame:  # type: ignore[type-arg]
    if not _DATA_PATH.exists():
        pytest.skip(
            f"Processed data not found: {_DATA_PATH}\n"
            "Run: uv run python scripts/prepare_data.py --scenario scenarios/puebla_yaz.yaml"
        )
    df = pd.read_csv(_DATA_PATH, parse_dates=["date"])
    start = pd.Timestamp(scenario["start_date"])
    end = pd.Timestamp(scenario["end_date"])
    df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
    if df.empty:
        pytest.skip("No data rows for scenario date window")
    return df.head(_SMOKE_ROUNDS)


@pytest.fixture(scope="module")
def llm() -> LLMClient:
    _skip_if_no_credentials()
    return LLMClient()


@pytest.fixture(scope="module")
def agents(scenario: dict, llm: LLMClient) -> list[HouseholdAgent]:
    result = []
    for agent_id, cfg in scenario["households"].items():
        result.append(
            HouseholdAgent(
                agent_id=agent_id,
                panel_area_m2=float(cfg["panel_area_m2"]),
                battery_capacity_kwh=float(cfg["battery_capacity_kwh"]),
                persona=str(cfg["persona"]),
                llm_client=llm,
            )
        )
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_single_llm_call(llm: LLMClient) -> None:
    """A single structured LLM call returns a valid AgentAction."""
    prompt = (
        "You manage household H1. Consumption today: 3.5 kWh, PV: 2.0 kWh, "
        "battery available: 2.0 kWh, grid capacity: 20 kWh. "
        "Respond with a JSON matching the schema."
    )
    action = llm.generate(prompt, AgentAction)

    assert isinstance(action, AgentAction)
    assert action.draw_kwh >= 0
    assert action.offer_kwh >= 0
    assert action.store_kwh >= 0
    assert len(action.reasoning) > 0


@pytest.mark.integration
def test_two_round_smoke(
    scenario: dict,  # type: ignore[type-arg]
    smoke_df: pd.DataFrame,
    agents: list[HouseholdAgent],
    llm: LLMClient,
    tmp_path: Path,
) -> None:
    """Full pipeline: 2 rounds, 3 agents — asserts structure, not values."""
    ref = scenario.get("referee") or {}
    params = {
        "grid_capacity_kwh": float(scenario["grid_capacity_kwh"]),
        "unit_utility": float(ref.get("unit_utility", 1.0)),
        "share_bonus": float(ref.get("share_bonus", 0.5)),
        "penalty_multiplier": float(ref.get("penalty_multiplier", 3.0)),
        "num_agents": len(scenario["households"]),
        "season_hint": str(scenario.get("season_hint", "")),
        "total_rounds": len(smoke_df),
    }

    logger = Logger(tmp_path / "smoke_run")
    logger.write_config(scenario, {"smoke": True})
    engine = RoundEngine(agents, params, logger)

    results: list[RoundResult] = []
    for i, (_, row) in enumerate(smoke_df.iterrows(), start=1):
        result = engine.run_round(i, row)
        results.append(result)

    # --- structural assertions ---
    assert len(results) == _SMOKE_ROUNDS

    for result in results:
        assert isinstance(result, RoundResult)
        assert len(result.payoffs) == len(agents)
        assert result.total_draw_kwh >= 0
        assert result.overflow_kwh >= 0
        assert result.overflow_kwh == pytest.approx(
            max(0.0, result.total_draw_kwh - result.capacity_kwh), abs=1e-6
        )
        for payoff in result.payoffs:
            assert payoff.draw_kwh >= 0
            assert payoff.offer_kwh >= 0

    # --- metrics ---
    agent_states = engine.agent_states
    assert len(agent_states) == len(agents)

    pv_total = sum(smoke_df[f"{a.agent_id.lower()}_pv_kwh"].sum() for a in agents)
    consumption_total = sum(smoke_df[f"{a.agent_id.lower()}_kwh"].sum() for a in agents)
    metrics = compute_run_metrics(results, agent_states, pv_total, consumption_total)

    assert metrics.total_welfare is not None
    assert 0.0 <= metrics.gini_coefficient <= 1.0
    assert metrics.capacity_violation_count >= 0

    logger.write_results_csv()
    logger.write_metrics_json(metrics)
    logger.close()

    # output files created
    run_dir = tmp_path / "smoke_run"
    assert (run_dir / "results.csv").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "log.jsonl").exists()

    print(
        f"\n[smoke] welfare={metrics.total_welfare:.2f}  "
        f"violations={metrics.capacity_violation_count}  "
        f"gini={metrics.gini_coefficient:.3f}  "
        f"tokens={llm.total_prompt_tokens}+{llm.total_completion_tokens}  "
        f"cost=${llm.estimated_cost_usd:.4f}"
    )
