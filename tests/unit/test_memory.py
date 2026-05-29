from core.memory import MemoryManager
from core.schemas import AgentAction, AgentPayoff


def _action(draw: float = 3.0) -> AgentAction:
    return AgentAction(draw_kwh=draw, offer_kwh=1.0, store_kwh=0.0, reasoning="test")


def _payoff(agent_id: str = "H1", net: float = 2.0) -> AgentPayoff:
    return AgentPayoff(
        agent_id=agent_id,
        draw_kwh=3.0,
        offer_kwh=1.0,
        gross_utility=3.0,
        share_bonus=0.5,
        penalty=1.5,
        net_payoff=net,
    )


def test_history_trimmed_to_k():
    mem = MemoryManager(agent_id="H1", k=3)
    for i in range(5):
        mem.update(
            round_num=i + 1,
            action=_action(),
            payoff=_payoff(),
            neighbor_actions={"H6": _action(2.0)},
        )
    assert len(mem._own_history) == 3
    # oldest rounds dropped, only rounds 3,4,5 remain
    assert mem._own_history[0][0] == 3


def test_summarize_contains_round_info():
    mem = MemoryManager(agent_id="H1")
    mem.update(
        round_num=1,
        action=_action(draw=4.0),
        payoff=_payoff(net=3.5),
        neighbor_actions={},
    )
    summary = mem.summarize()
    assert "Round 1" in summary
    assert "4.00" in summary
    assert "3.50" in summary


def test_neighbor_summary_generated():
    mem = MemoryManager(agent_id="H1")
    mem.update(
        round_num=1,
        action=_action(),
        payoff=_payoff(),
        neighbor_actions={"H6": _action(2.5), "H8": _action(5.0)},
    )
    assert "H6" in mem.neighbor_summaries
    assert "H8" in mem.neighbor_summaries
    summary = mem.summarize()
    assert "H6" in summary
    assert "H8" in summary


def test_empty_memory_summarize():
    mem = MemoryManager(agent_id="H1")
    assert mem.summarize() == "No history yet."
