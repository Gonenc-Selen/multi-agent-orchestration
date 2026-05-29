"""CLI entry point. Usage:

    uv run python -m core.run --scenario scenarios/puebla_yaz.yaml
    uv run python -m core.run --scenario scenarios/puebla_yaz.yaml --runs 5
    uv run python -m core.run --scenario scenarios/puebla_yaz.yaml --smoke
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from core.agent import HouseholdAgent
from core.llm_client import llm_client
from core.logger import Logger
from core.metrics import compute_run_metrics
from core.round_engine import RoundEngine
from core.schemas import ScenarioConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_SMOKE_ROUNDS = 2
_DATA_PATH = Path("data/processed/community_window.csv")


def _load_scenario(path: Path) -> ScenarioConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ScenarioConfig.model_validate(raw)


def _build_agents(scenario: ScenarioConfig) -> list[HouseholdAgent]:
    agents = []
    tolerance = scenario.promise_keeping.tolerance_kwh
    for agent_id, cfg in scenario.households.items():
        agents.append(
            HouseholdAgent(
                agent_id=agent_id,
                panel_area_m2=cfg.panel_area_m2,
                battery_capacity_kwh=cfg.battery_capacity_kwh,
                persona=cfg.persona,
                llm_client=llm_client,
                tolerance_kwh=tolerance,
            )
        )
    return agents


def _build_scenario_params(scenario: ScenarioConfig) -> dict[str, Any]:
    ref = scenario.referee
    return {
        "grid_capacity_kwh": scenario.grid_capacity_kwh,
        "unit_utility": ref.unit_utility,
        "share_bonus": ref.share_bonus,
        "penalty_multiplier": ref.penalty_multiplier,
        "num_agents": len(scenario.households),
        "season_hint": scenario.season_hint,
        "communication_mode": scenario.communication_mode,
        "total_rounds": 0,  # set after data loaded
    }


def _load_data(scenario: ScenarioConfig) -> pd.DataFrame:
    if not _DATA_PATH.exists():
        log.error(
            "Processed data not found: %s\n"
            "Run first: uv run python scripts/prepare_data.py --scenario <yaml>",
            _DATA_PATH,
        )
        sys.exit(1)

    df = pd.read_csv(_DATA_PATH, parse_dates=["date"])
    start = pd.Timestamp(scenario.start_date)
    end = pd.Timestamp(scenario.end_date)
    df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)

    if df.empty:
        log.error("No data rows for %s -> %s", start.date(), end.date())
        sys.exit(1)

    return df


def _print_promise_summary(
    metrics: "RunMetrics",  # type: ignore[name-defined]  # noqa: F821
    intent_log: dict,
    results: list,
    tolerance_kwh: float,
) -> None:
    """Print V2 promise-keeping summary to console."""
    agent_rates = {
        aid: m.promise_kept_rate
        for aid, m in metrics.agent_metrics.items()
    }
    avg_rate = sum(agent_rates.values()) / len(agent_rates) if agent_rates else 0.0
    least_faithful = min(agent_rates, key=lambda a: agent_rates[a])

    # Pearson correlation: intent_draw vs actual_draw across all agents & rounds
    intent_draws: list[float] = []
    actual_draws: list[float] = []
    actual_by_round = {
        r.round_num: {p.agent_id: p.draw_kwh for p in r.payoffs}
        for r in results
    }
    for round_num, agent_intents in intent_log.items():
        actuals = actual_by_round.get(round_num, {})
        for aid, intent in agent_intents.items():
            if aid in actuals:
                intent_draws.append(intent.intent_draw_kwh)
                actual_draws.append(actuals[aid])

    if len(intent_draws) >= 2:
        corr = float(np.corrcoef(intent_draws, actual_draws)[0, 1])
    else:
        corr = float("nan")

    print("\n=== V2 Promise-Keeping Summary ===")
    print(f"Avg promise_kept_rate (community): {avg_rate:.2f}")
    print(f"Least faithful agent             : {least_faithful} (rate: {agent_rates[least_faithful]:.2f})")
    print(f"Intent-action correlation        : r={corr:.2f}  (Pearson, intent_draw vs actual_draw)")
    print(f"Tolerance threshold              : {tolerance_kwh} kWh")


def run_once(scenario_path: Path, smoke: bool = False) -> None:
    scenario = _load_scenario(scenario_path)
    df = _load_data(scenario)

    if smoke:
        df = df.head(_SMOKE_ROUNDS)
        log.info("Smoke mode: running %d rounds", len(df))

    params = _build_scenario_params(scenario)
    params["total_rounds"] = len(df)

    run_dir = Path("runs") / datetime.now().strftime("%Y%m%dT%H%M%S")
    logger = Logger(run_dir)
    logger.write_config(scenario.model_dump(), {"scenario_path": str(scenario_path)})

    agents = _build_agents(scenario)
    engine = RoundEngine(agents, params, logger)

    log.info("Starting run: %d agents, %d rounds", len(agents), len(df))
    results = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        log.info("Round %d/%d ...", i, len(df))
        result = engine.run_round(i, row)
        results.append(result)
        log.info(
            "  overflow=%.2f kWh  payoffs=%s",
            result.overflow_kwh,
            {p.agent_id: round(p.net_payoff, 2) for p in result.payoffs},
        )

    agent_states = engine.agent_states
    pv_total = sum(df[f"{a.agent_id.lower()}_pv_kwh"].sum() for a in agents)
    consumption_total = sum(df[f"{a.agent_id.lower()}_kwh"].sum() for a in agents)

    metrics = compute_run_metrics(
        results,
        agent_states,
        pv_total,
        consumption_total,
        intent_log=engine.intent_log if engine.intent_log else None,
        tolerance_kwh=scenario.promise_keeping.tolerance_kwh,
        communication_mode=scenario.communication_mode,
    )

    logger.write_results_csv()
    logger.write_metrics_json(metrics)
    logger.close()

    log.info("Run complete -> %s", run_dir)
    log.info(
        "Total welfare=%.2f  violations=%d  gini=%.3f  cost=$%.4f",
        metrics.total_welfare,
        metrics.capacity_violation_count,
        metrics.gini_coefficient,
        llm_client.estimated_cost_usd,
    )

    if scenario.communication_mode == "v2" and engine.intent_log:
        _print_promise_summary(
            metrics,
            engine.intent_log,
            results,
            scenario.promise_keeping.tolerance_kwh,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic Energy CPR runner")
    parser.add_argument("--scenario", required=True, type=Path, help="Path to scenario YAML")
    parser.add_argument("--runs", type=int, default=1, help="Number of independent runs")
    parser.add_argument("--smoke", action="store_true", help=f"Run only {_SMOKE_ROUNDS} rounds")
    args = parser.parse_args()

    for i in range(args.runs):
        if args.runs > 1:
            log.info("=== Run %d/%d ===", i + 1, args.runs)
        run_once(args.scenario, smoke=args.smoke)


if __name__ == "__main__":
    main()
