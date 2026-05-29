"""Append a scenario's date window to community_window.csv without overwriting.

Already-present dates are skipped (idempotent). Useful when multiple scenario
windows need to coexist in the same processed file.

Usage:
    uv run python scripts/append_window.py --scenario scenarios/puebla_ilkbahar.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from core.pv_model import daily_pv_kwh
from core.schemas import ScenarioConfig

_DATA_DIR = Path("Data")
_OUT_PATH = Path("data/processed/community_window.csv")

_WEATHER_FILE = "weather_outdoor variables.csv"


def load_household(filename: str) -> pd.DataFrame:
    path = _DATA_DIR / filename
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["Time"], dayfirst=True).dt.normalize()
    df = df.rename(columns={"kWh": "kwh"})
    return df[["date", "kwh"]].dropna()


def load_weather_daily() -> pd.DataFrame:
    path = _DATA_DIR / _WEATHER_FILE
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["Time"], dayfirst=True).dt.normalize()
    solar = (
        df.groupby("date")["Solar Radiation (W/m^2)"]
        .mean()
        .reset_index()
        .rename(columns={"Solar Radiation (W/m^2)": "solar_w_m2"})
    )
    return solar


def append_window(scenario_path: Path) -> None:
    with open(scenario_path, encoding="utf-8") as f:
        scenario = ScenarioConfig.model_validate(yaml.safe_load(f))

    start = pd.Timestamp(scenario.start_date)
    end = pd.Timestamp(scenario.end_date)
    households = scenario.households

    # Load existing output if present
    existing_dates: set[pd.Timestamp] = set()
    if _OUT_PATH.exists():
        existing = pd.read_csv(_OUT_PATH, parse_dates=["date"])
        existing_dates = set(existing["date"])

    weather = load_weather_daily()

    merged: pd.DataFrame | None = None
    for agent_id, cfg in households.items():
        filename = cfg.csv + ".csv"
        hdf = load_household(filename)
        col = agent_id.lower()
        hdf = hdf.rename(columns={"kwh": f"{col}_kwh"})
        merged = hdf if merged is None else merged.merge(hdf, on="date", how="inner")

    assert merged is not None
    merged = merged.merge(weather, on="date", how="inner")
    merged = merged[
        (merged["date"] >= start) & (merged["date"] <= end)
    ].reset_index(drop=True)

    if merged.empty:
        print(f"ERROR: No data for {start.date()} to {end.date()}")
        sys.exit(1)

    # Drop dates already in the file
    new_rows = merged[~merged["date"].isin(existing_dates)].copy()
    if new_rows.empty:
        print(f"All dates for {start.date()}–{end.date()} already present. Nothing to do.")
        return

    # Compute PV using scenario PV params
    pv_efficiency = scenario.pv.efficiency
    pv_daylight_hours = scenario.pv.daylight_hours
    for agent_id, cfg in households.items():
        col = agent_id.lower()
        area = cfg.panel_area_m2
        new_rows[f"{col}_pv_kwh"] = new_rows["solar_w_m2"].apply(
            lambda r, a=area: daily_pv_kwh(r, a, pv_efficiency, pv_daylight_hours)
        )

    # Reorder columns
    cols = ["date"]
    for agent_id in households:
        col = agent_id.lower()
        cols += [f"{col}_kwh", f"{col}_pv_kwh"]
    cols.append("solar_w_m2")
    new_rows = new_rows[cols]

    # Append to file
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = not _OUT_PATH.exists()
    new_rows.to_csv(_OUT_PATH, mode="a", index=False, header=header)
    print(f"Appended {len(new_rows)} new rows to {_OUT_PATH}")
    print(new_rows.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Append scenario window to community_window.csv")
    parser.add_argument("--scenario", required=True, type=Path)
    args = parser.parse_args()
    append_window(args.scenario)


if __name__ == "__main__":
    main()
