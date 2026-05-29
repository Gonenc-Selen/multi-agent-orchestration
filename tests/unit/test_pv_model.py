import pytest

from core.pv_model import daily_pv_kwh


def test_known_output():
    # 600 W/m2 * 25 m2 * 0.18 * 10h / 1000 = 27.0 kWh
    result = daily_pv_kwh(radiation_w_m2=600.0, panel_area_m2=25.0)
    assert result == pytest.approx(27.0)


def test_zero_radiation():
    assert daily_pv_kwh(radiation_w_m2=0.0, panel_area_m2=25.0) == 0.0


def test_zero_area():
    assert daily_pv_kwh(radiation_w_m2=600.0, panel_area_m2=0.0) == 0.0


def test_custom_efficiency():
    # 500 * 10 * 0.20 * 10 / 1000 = 10.0
    result = daily_pv_kwh(radiation_w_m2=500.0, panel_area_m2=10.0, efficiency=0.20)
    assert result == pytest.approx(10.0)


def test_custom_daylight_hours():
    # 400 * 20 * 0.18 * 8 / 1000 = 11.52
    result = daily_pv_kwh(radiation_w_m2=400.0, panel_area_m2=20.0, daylight_hours=8.0)
    assert result == pytest.approx(11.52)
