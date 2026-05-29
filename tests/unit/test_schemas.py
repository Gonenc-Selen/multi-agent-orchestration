import pytest
from pydantic import ValidationError

from core.schemas import AgentAction


def test_valid_action():
    a = AgentAction(draw_kwh=3.0, offer_kwh=1.0, store_kwh=0.5, reasoning="ok")
    assert a.draw_kwh == 3.0


def test_negative_draw_rejected():
    with pytest.raises(ValidationError, match="draw_kwh"):
        AgentAction(draw_kwh=-1.0, offer_kwh=0.0, store_kwh=0.0, reasoning="x")


def test_negative_offer_rejected():
    with pytest.raises(ValidationError, match="offer_kwh"):
        AgentAction(draw_kwh=0.0, offer_kwh=-0.5, store_kwh=0.0, reasoning="x")


def test_negative_store_rejected():
    with pytest.raises(ValidationError, match="store_kwh"):
        AgentAction(draw_kwh=0.0, offer_kwh=0.0, store_kwh=-1.0, reasoning="x")


def test_empty_reasoning_rejected():
    with pytest.raises(ValidationError, match="reasoning"):
        AgentAction(draw_kwh=0.0, offer_kwh=0.0, store_kwh=0.0, reasoning="   ")


def test_zero_values_allowed():
    a = AgentAction(draw_kwh=0.0, offer_kwh=0.0, store_kwh=0.0, reasoning="idle")
    assert a.draw_kwh == 0.0
