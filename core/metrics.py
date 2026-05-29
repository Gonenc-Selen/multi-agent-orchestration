from core.schemas import (
    AgentIndividualMetrics,
    AgentIntent,
    AgentState,
    RoundResult,
    RunMetrics,
)


def gini_coefficient(values: list[float]) -> float:
    """Gini coefficient for a list of values. Returns 0.0 for empty or all-zero input."""
    n = len(values)
    if n == 0 or sum(values) == 0:
        return 0.0
    sorted_v = sorted(values)
    cumsum = 0.0
    for i, v in enumerate(sorted_v):
        cumsum += (2 * (i + 1) - n - 1) * v
    return cumsum / (n * sum(sorted_v))


def self_sufficiency_ratio(total_pv_kwh: float, total_consumption_kwh: float) -> float:
    """Fraction of consumption covered by local PV generation."""
    if total_consumption_kwh == 0:
        return 0.0
    return min(1.0, total_pv_kwh / total_consumption_kwh)


def capacity_violations(results: list[RoundResult]) -> tuple[int, float]:
    """Return (violation_count, avg_overflow_kwh) across all rounds."""
    violations = [r.overflow_kwh for r in results if r.overflow_kwh > 0]
    count = len(violations)
    avg = sum(violations) / count if count else 0.0
    return count, avg


def compute_run_metrics(
    results: list[RoundResult],
    agent_states: list[AgentState],
    pv_total: float,
    consumption_total: float,
    intent_log: dict[int, dict[str, AgentIntent]] | None = None,
    tolerance_kwh: float = 0.5,
    communication_mode: str = "v1",
) -> RunMetrics:
    """Aggregate KPIs for a completed run."""
    violation_count, violation_avg = capacity_violations(results)

    net_payoffs = [s.total_net_payoff for s in agent_states]
    total_welfare = sum(net_payoffs)
    gini = gini_coefficient(net_payoffs)
    ss_ratio = self_sufficiency_ratio(pv_total, consumption_total)

    # promise_kept_rate: per-agent, based on intent vs actual draw
    promise_kept: dict[str, list[bool]] = {s.agent_id: [] for s in agent_states}
    if intent_log:
        actual_by_round: dict[int, dict[str, float]] = {
            r.round_num: {p.agent_id: p.draw_kwh for p in r.payoffs}
            for r in results
        }
        for round_num, agent_intents in intent_log.items():
            actuals = actual_by_round.get(round_num, {})
            for aid, intent in agent_intents.items():
                if aid in actuals:
                    kept = abs(intent.intent_draw_kwh - actuals[aid]) < tolerance_kwh
                    promise_kept[aid].append(kept)

    agent_metrics: dict[str, AgentIndividualMetrics] = {}
    for state in agent_states:
        aid = state.agent_id
        total_offered = sum(
            p.offer_kwh
            for r in results
            for p in r.payoffs
            if p.agent_id == aid
        )
        violation_contrib = sum(
            p.draw_kwh / r.total_draw_kwh * r.overflow_kwh
            for r in results
            if r.overflow_kwh > 0 and r.total_draw_kwh > 0
            for p in r.payoffs
            if p.agent_id == aid
        )
        kept_list = promise_kept[aid]
        pkr = sum(kept_list) / len(kept_list) if kept_list else 1.0
        agent_metrics[aid] = AgentIndividualMetrics(
            net_profit=state.total_net_payoff,
            total_offered_kwh=total_offered,
            violation_contribution_kwh=violation_contrib,
            promise_kept_rate=pkr,
        )

    return RunMetrics(
        total_welfare=total_welfare,
        capacity_violation_count=violation_count,
        capacity_violation_avg_kwh=violation_avg,
        gini_coefficient=gini,
        self_sufficiency_ratio=ss_ratio,
        agent_metrics=agent_metrics,
        communication_mode=communication_mode,
    )
