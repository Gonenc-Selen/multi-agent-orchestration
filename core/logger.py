from __future__ import annotations

import csv
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.schemas import AgentAction, RoundResult, RunMetrics

log = logging.getLogger(__name__)


class Logger:
    """Writes log.jsonl, results.csv, metrics.json, and config.yaml for a run."""

    def __init__(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self._run_dir = run_dir
        self._jsonl = open(run_dir / "log.jsonl", "w", encoding="utf-8")
        self._round_rows: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Per-event logging
    # ------------------------------------------------------------------

    def log_action(
        self,
        round_num: int,
        agent_id: str,
        action: AgentAction,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        self._write(
            {
                "event": "agent_action",
                "ts": _now(),
                "round": round_num,
                "agent_id": agent_id,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                **action.model_dump(),
            }
        )

    def log_round_result(self, round_num: int, result: RoundResult) -> None:
        data = result.model_dump()
        data.update({"event": "round_result", "ts": _now()})
        self._write(data)

        row: dict[str, Any] = {
            "round_num": round_num,
            "total_draw_kwh": result.total_draw_kwh,
            "capacity_kwh": result.capacity_kwh,
            "overflow_kwh": result.overflow_kwh,
        }
        for p in result.payoffs:
            row[f"{p.agent_id.lower()}_draw_kwh"] = p.draw_kwh
            row[f"{p.agent_id.lower()}_offer_kwh"] = p.offer_kwh
            row[f"{p.agent_id.lower()}_net_payoff"] = p.net_payoff
            row[f"{p.agent_id.lower()}_penalty"] = p.penalty
        self._round_rows.append(row)

    def log_battery_clip(
        self, round_num: int, agent_id: str, requested: float, actual: float
    ) -> None:
        self._write(
            {
                "event": "battery_clip",
                "ts": _now(),
                "round": round_num,
                "agent_id": agent_id,
                "requested_kwh": requested,
                "actual_kwh": actual,
            }
        )

    # ------------------------------------------------------------------
    # End-of-run outputs
    # ------------------------------------------------------------------

    def write_results_csv(self) -> None:
        if not self._round_rows:
            return
        path = self._run_dir / "results.csv"
        fieldnames = list(self._round_rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._round_rows)
        log.info("results.csv written → %s", path)

    def write_metrics_json(self, metrics: RunMetrics) -> None:
        path = self._run_dir / "metrics.json"
        path.write_text(
            json.dumps(metrics.model_dump(), indent=2), encoding="utf-8"
        )
        log.info("metrics.json written → %s", path)

    def write_config(self, scenario: dict[str, Any], extra: dict[str, Any]) -> None:
        import yaml  # local import — only needed here

        path = self._run_dir / "config.yaml"
        payload = {"scenario": scenario, **extra}
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(payload, f, allow_unicode=True, sort_keys=False)
        log.info("config.yaml written → %s", path)

    def close(self) -> None:
        self._jsonl.close()

    # ------------------------------------------------------------------

    def _write(self, data: dict[str, Any]) -> None:
        self._jsonl.write(json.dumps(data, default=str) + "\n")
        self._jsonl.flush()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
