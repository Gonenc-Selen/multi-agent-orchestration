from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml


AGENT_IDS = ["H1", "H6", "H8"]


@dataclass
class RoundBundle:
    round_num: int
    intents: dict[str, dict] = field(default_factory=dict)
    neg_rounds: dict[int, list[dict]] = field(default_factory=dict)
    actions: dict[str, dict] = field(default_factory=dict)
    result: dict | None = None


@dataclass
class RunData:
    path: Path
    config: dict
    metrics: dict
    results_df: pd.DataFrame
    events: list[dict]
    rounds: dict[int, RoundBundle]
    mode: str
    tolerance_kwh: float
    scenario_name: str
    timestamp_str: str


def get_tolerance(config: dict) -> float:
    return (
        config.get("scenario", {})
        .get("promise_keeping", {})
        .get("tolerance_kwh", 0.5)
    )


def is_promise_kept(intent: dict, action: dict, tolerance: float) -> bool:
    return abs(intent["intent_draw_kwh"] - action["draw_kwh"]) < tolerance


def get_cumulative(run: "RunData", up_to_round: int) -> dict[str, dict]:
    """Per-agent cumulative stats from round 1 to up_to_round.

    Tolerance priority: metrics.json → config.yaml (run.tolerance_kwh) → 0.5
    """
    tol: float = run.metrics.get("tolerance_kwh") or run.tolerance_kwh or 0.5
    n_total = max(run.rounds.keys(), default=10)

    agg: dict[str, dict] = {
        aid: {
            "total_draw": 0.0,
            "total_penalty": 0.0,
            "total_net_payoff": 0.0,
            "promise_states": [],   # "kept"|"broken"|"no_intent"|"future" per round 1..n_total
            "neg_message_count": 0,
        }
        for aid in AGENT_IDS
    }

    for rn in range(1, n_total + 1):
        bundle = run.rounds.get(rn)
        for aid in AGENT_IDS:
            if rn > up_to_round or bundle is None:
                agg[aid]["promise_states"].append("future")
                continue

            action = bundle.actions.get(aid, {})
            agg[aid]["total_draw"] += action.get("draw_kwh", 0.0)

            if bundle.result:
                for p in bundle.result.get("payoffs", []):
                    if p.get("agent_id") == aid:
                        agg[aid]["total_penalty"] += p.get("penalty", 0.0)
                        agg[aid]["total_net_payoff"] += p.get("net_payoff", 0.0)

            intent = bundle.intents.get(aid)
            if intent and action:
                kept = is_promise_kept(intent, action, tol)
                agg[aid]["promise_states"].append("kept" if kept else "broken")
            else:
                agg[aid]["promise_states"].append("no_intent")

            for msgs in bundle.neg_rounds.values():
                for msg in msgs:
                    if msg.get("from_agent") == aid:
                        agg[aid]["neg_message_count"] += 1

    return agg


def group_events(events: list[dict]) -> dict[int, RoundBundle]:
    rounds: dict[int, RoundBundle] = {}

    for event in events:
        ev_type = event.get("event", "")
        round_num = event.get("round") or event.get("round_num")
        if round_num is None:
            continue

        if round_num not in rounds:
            rounds[round_num] = RoundBundle(round_num=round_num)
        bundle = rounds[round_num]

        if ev_type == "agent_intent":
            bundle.intents[event["agent_id"]] = event
        elif ev_type == "negotiation_message":
            neg_r = event.get("negotiation_round", 1)
            if neg_r not in bundle.neg_rounds:
                bundle.neg_rounds[neg_r] = []
            bundle.neg_rounds[neg_r].append(event)
        elif ev_type == "agent_action":
            bundle.actions[event["agent_id"]] = event
        elif ev_type == "round_result":
            bundle.result = event

    return rounds


def load_run_data(run_path: Path) -> RunData:
    config: dict = {}
    config_path = run_path / "config.yaml"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    metrics: dict = {}
    metrics_path = run_path / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)

    results_df = pd.DataFrame()
    results_path = run_path / "results.csv"
    if results_path.exists():
        results_df = pd.read_csv(results_path)

    events: list[dict] = []
    log_path = run_path / "log.jsonl"
    if log_path.exists():
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    rounds = group_events(events)
    scenario_cfg = config.get("scenario", {})
    mode = scenario_cfg.get("communication_mode", metrics.get("communication_mode", "v1"))
    tolerance = get_tolerance(config)
    scenario_name = scenario_cfg.get("name", "unknown")

    return RunData(
        path=run_path,
        config=config,
        metrics=metrics,
        results_df=results_df,
        events=events,
        rounds=rounds,
        mode=mode,
        tolerance_kwh=tolerance,
        scenario_name=scenario_name,
        timestamp_str=run_path.name,
    )


def load_run_by_id(run_id: str, runs_dir: Path) -> "RunData | None":
    path = runs_dir / run_id
    if not path.exists():
        return None
    try:
        return load_run_data(path)
    except Exception:
        return None


def list_all_runs(runs_dir: Path) -> list[RunData]:
    runs: list[RunData] = []
    if not runs_dir.exists():
        return runs
    for d in sorted(runs_dir.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        if not (d / "metrics.json").exists():
            continue
        try:
            runs.append(load_run_data(d))
        except Exception:
            pass
    return runs


def filter_runs_by_mode(runs: list[RunData], mode: str) -> list[RunData]:
    return [r for r in runs if r.mode == mode]
