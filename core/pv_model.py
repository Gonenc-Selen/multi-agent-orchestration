def daily_pv_kwh(
    radiation_w_m2: float,
    panel_area_m2: float,
    efficiency: float = 0.18,
    daylight_hours: float = 10.0,
) -> float:
    """Deterministic PV output — no ML, physical formula only."""
    return radiation_w_m2 * panel_area_m2 * efficiency * daylight_hours / 1000.0
