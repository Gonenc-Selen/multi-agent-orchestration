import pytest

from core.referee import compute_payoffs
from core.schemas import AgentAction


def _actions(**kwargs: tuple[float, float, float]) -> dict[str, AgentAction]:
    return {
        aid: AgentAction(draw_kwh=d, offer_kwh=o, store_kwh=s, reasoning="test")
        for aid, (d, o, s) in kwargs.items()
    }


def test_no_overflow_no_penalty():
    actions = _actions(H1=(4.0, 0.0, 0.0), H6=(2.0, 0.0, 0.0), H8=(5.0, 0.0, 0.0))
    result = compute_payoffs(actions, round_num=1, capacity_kwh=20.0)
    assert result.overflow_kwh == 0.0
    for p in result.payoffs:
        assert p.penalty == 0.0


def test_overflow_proportional_penalty():
    # total draw = 15, capacity = 10 → overflow = 5
    actions = _actions(H1=(5.0, 0.0, 0.0), H6=(5.0, 0.0, 0.0), H8=(5.0, 0.0, 0.0))
    result = compute_payoffs(actions, round_num=1, capacity_kwh=10.0, penalty_multiplier=3.0)
    assert result.overflow_kwh == pytest.approx(5.0)
    # each agent drew 1/3 → penalty = 5 * (1/3) * 3 = 5.0
    for p in result.payoffs:
        assert p.penalty == pytest.approx(5.0)


def test_unequal_draw_proportional_penalty():
    # H1 draws 8, H6 draws 2, capacity = 5 → overflow = 5
    actions = _actions(H1=(8.0, 0.0, 0.0), H6=(2.0, 0.0, 0.0))
    result = compute_payoffs(actions, round_num=1, capacity_kwh=5.0, penalty_multiplier=1.0)
    assert result.overflow_kwh == pytest.approx(5.0)
    h1 = next(p for p in result.payoffs if p.agent_id == "H1")
    h6 = next(p for p in result.payoffs if p.agent_id == "H6")
    # H1 penalty = 5 * 0.8 * 1 = 4.0
    assert h1.penalty == pytest.approx(4.0)
    # H6 penalty = 5 * 0.2 * 1 = 1.0
    assert h6.penalty == pytest.approx(1.0)


def test_offer_bonus():
    actions = _actions(H1=(2.0, 3.0, 0.0))
    result = compute_payoffs(actions, round_num=1, capacity_kwh=20.0, share_bonus=0.5)
    h1 = result.payoffs[0]
    assert h1.share_bonus == pytest.approx(1.5)


def test_all_zero_draw():
    actions = _actions(H1=(0.0, 0.0, 0.0), H6=(0.0, 0.0, 0.0))
    result = compute_payoffs(actions, round_num=1, capacity_kwh=10.0)
    assert result.overflow_kwh == 0.0
    for p in result.payoffs:
        assert p.penalty == 0.0
        assert p.net_payoff == 0.0
