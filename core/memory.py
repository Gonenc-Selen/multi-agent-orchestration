from __future__ import annotations

from core.schemas import AgentAction, AgentIntent, AgentPayoff

_DEFAULT_K = 3


class MemoryManager:
    """Sliding window memory for a single agent. No LLM calls — template-based summaries."""

    def __init__(
        self, agent_id: str, k: int = _DEFAULT_K, tolerance_kwh: float = 0.5
    ) -> None:
        self.agent_id = agent_id
        self.k = k
        self._tolerance_kwh = tolerance_kwh
        self._own_history: list[tuple[int, AgentAction, AgentPayoff]] = []
        self.neighbor_summaries: dict[str, str] = {}
        # (round_num, intent_draw | None, actual_draw)
        self._neighbor_history: dict[str, list[tuple[int, float | None, float]]] = {}

    def update(
        self,
        round_num: int,
        action: AgentAction,
        payoff: AgentPayoff,
        neighbor_actions: dict[str, AgentAction],
        neighbor_intents: dict[str, AgentIntent] | None = None,
    ) -> None:
        """Record this round's outcome and trim to K window."""
        self._own_history.append((round_num, action, payoff))
        if len(self._own_history) > self.k:
            self._own_history = self._own_history[-self.k :]

        for nid, naction in neighbor_actions.items():
            if nid not in self._neighbor_history:
                self._neighbor_history[nid] = []
            intent_draw: float | None = None
            if neighbor_intents and nid in neighbor_intents:
                intent_draw = neighbor_intents[nid].intent_draw_kwh
            self._neighbor_history[nid].append((round_num, intent_draw, naction.draw_kwh))
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
            actual_draws = [a for _, _, a in history]
            intent_draws = [i for _, i, _ in history if i is not None]
            avg_actual = sum(actual_draws) / len(actual_draws)

            if intent_draws and len(intent_draws) == len(actual_draws):
                avg_intent = sum(intent_draws) / len(intent_draws)
                avg_gap = sum(
                    abs(a - i) for i, a in zip(intent_draws, actual_draws)
                ) / len(intent_draws)
                faithfulness = (
                    "yüksek" if avg_gap <= self._tolerance_kwh else "düşük"
                )
                self.neighbor_summaries[nid] = (
                    f"avg draw {avg_actual:.2f} kWh, avg intent {avg_intent:.2f} kWh"
                    f" — söz tutma oranı {faithfulness}"
                    f" (ort. sapma {avg_gap:.2f} kWh, eşik {self._tolerance_kwh} kWh)"
                )
            else:
                self.neighbor_summaries[nid] = (
                    f"avg draw {avg_actual:.2f} kWh over last {len(actual_draws)} round(s)"
                )
