"""Prepare community_window.csv from raw CSVs.

Usage:
    uv run python scripts/prepare_data.py --scenario scenarios/puebla_ilkbahar.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from core.pv_model import daily_pv_kwh
from core.schemas import ScenarioConfig

# Raw data paths
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
    # Daily mean solar radiation
    solar = (
        df.groupby("date")["Solar Radiation (W/m^2)"]
        .mean()
        .reset_index()
        .rename(columns={"Solar Radiation (W/m^2)": "solar_w_m2"})
    )
    return solar


def prepare(scenario_path: Path, append: bool = False) -> None:
    with open(scenario_path, encoding="utf-8") as f:
        scenario = ScenarioConfig.model_validate(yaml.safe_load(f))

    start = pd.Timestamp(scenario.start_date)
    end = pd.Timestamp(scenario.end_date)
    households = scenario.households

    # Load weather
    weather = load_weather_daily()

    # Merge all households
    merged: pd.DataFrame | None = None
    for agent_id, cfg in households.items():
        filename = cfg.csv + ".csv"
        hdf = load_household(filename)
        col = agent_id.lower()
        hdf = hdf.rename(columns={"kwh": f"{col}_kwh"})

        if merged is None:
            merged = hdf
        else:
            merged = merged.merge(hdf, on="date", how="inner")

    assert merged is not None
    merged = merged.merge(weather, on="date", how="inner")

    # Filter to scenario window
    merged = merged[
        (merged["date"] >= start) & (merged["date"] <= end)
    ].reset_index(drop=True)

    if merged.empty:
        print(f"ERROR: No data for {start.date()} to {end.date()}")
        sys.exit(1)

    # Compute PV per household using scenario PV params
    pv_efficiency = scenario.pv.efficiency
    pv_daylight_hours = scenario.pv.daylight_hours
    for agent_id, cfg in households.items():
        col = agent_id.lower()
        area = cfg.panel_area_m2
        merged[f"{col}_pv_kwh"] = merged["solar_w_m2"].apply(
            lambda r, a=area: daily_pv_kwh(r, a, pv_efficiency, pv_daylight_hours)
        )

    # Reorder columns
    cols = ["date"]
    for agent_id in households:
        col = agent_id.lower()
        cols += [f"{col}_kwh", f"{col}_pv_kwh"]
    cols.append("solar_w_m2")
    merged = merged[cols]

    # Write output
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if append and _OUT_PATH.exists():
        existing = pd.read_csv(_OUT_PATH, parse_dates=["date"])
        combined = pd.concat([existing, merged], ignore_index=True)
        # Drop duplicate dates keeping the newly added rows (last occurrence)
        combined = combined.drop_duplicates(subset="date", keep="last")
        combined = combined.sort_values("date").reset_index(drop=True)
        combined.to_csv(_OUT_PATH, index=False)
        print(f"Appended {len(merged)} rows -> {len(combined)} total rows in {_OUT_PATH}")
        print(merged.to_string(index=False))
    else:
        merged.to_csv(_OUT_PATH, index=False)
        print(f"Written {len(merged)} rows to {_OUT_PATH}")
        print(merged.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare community_window.csv")
    parser.add_argument("--scenario", required=True, type=Path)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing community_window.csv instead of overwriting",
    )
    args = parser.parse_args()
    prepare(args.scenario, append=args.append)


if __name__ == "__main__":
    main()
