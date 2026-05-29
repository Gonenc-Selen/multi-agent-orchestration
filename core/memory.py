from __future__ import annotations

from core.schemas import AgentAction, AgentPayoff

_DEFAULT_K = 3


class MemoryManager:
    """Sliding window memory for a single agent. No LLM calls — template-based summaries."""

    def __init__(self, agent_id: str, k: int = _DEFAULT_K) -> None:
        self.agent_id = agent_id
        self.k = k
        self._own_history: list[tuple[int, AgentAction, AgentPayoff]] = []
        self.neighbor_summaries: dict[str, str] = {}
        self._neighbor_history: dict[str, list[tuple[int, float]]] = {}

    def update(
        self,
        round_num: int,
        action: AgentAction,
        payoff: AgentPayoff,
        neighbor_actions: dict[str, AgentAction],
    ) -> None:
        """Record this round's outcome and trim to K window."""
        self._own_history.append((round_num, action, payoff))
        if len(self._own_history) > self.k:
            self._own_history = self._own_history[-self.k :]

        for nid, naction in neighbor_actions.items():
            if nid not in self._neighbor_history:
                self._neighbor_history[nid] = []
            self._neighbor_history[nid].append((round_num, naction.draw_kwh))
            if len(self._neighbor_history[nid]) > self.k:
                self._neighbor_history[nid] = self._neighbor_history[nid][-self.k :]

        self._refresh_neighbor_summaries()

    def summarize(self) -> str:
        """Return a ≤200-token template-based summary of own history."""
        if not self._own_history:
            return "No history yet."

        lines: list[str] = []
        for round_num, action, payoff in self._own_history:
            lines.append(
                f"Round {round_num}: drew {action.draw_kwh:.2f} kWh, "
                f"offered {action.offer_kwh:.2f} kWh, "
                f"net payoff {payoff.net_payoff:.2f}."
            )

        neighbor_lines: list[str] = []
        for nid, summary in self.neighbor_summaries.items():
            neighbor_lines.append(f"  {nid}: {summary}")

        parts = [f"Own history (last {len(self._own_history)} rounds):"]
        parts.extend("  " + line for line in lines)
        if neighbor_lines:
            parts.append("Neighbor observations:")
            parts.extend(neighbor_lines)

        return "\n".join(parts)

    def _refresh_neighbor_summaries(self) -> None:
        for nid, history in self._neighbor_history.items():
            if not history:
                continue
            draws = [d for _, d in history]
            avg = sum(draws) / len(draws)
            self.neighbor_summaries[nid] = (
                f"avg draw {avg:.2f} kWh over last {len(draws)} round(s)"
            )
