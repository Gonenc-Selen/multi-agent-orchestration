from core.schemas import AgentAction, AgentPayoff, RoundResult


def compute_payoffs(
    actions: dict[str, AgentAction],
    round_num: int,
    capacity_kwh: float,
    unit_utility: float = 1.0,
    share_bonus: float = 0.5,
    penalty_multiplier: float = 3.0,
) -> RoundResult:
    """Soft proportional penalty CPR payoff calculation.

    Each agent's penalty scales with their share of total draw.
    """
    total_draw = sum(a.draw_kwh for a in actions.values())
    overflow = max(0.0, total_draw - capacity_kwh)

    payoffs: list[AgentPayoff] = []
    for agent_id, action in actions.items():
        gross = action.draw_kwh * unit_utility
        bonus = action.offer_kwh * share_bonus

        if total_draw > 0:
            pay_share = action.draw_kwh / total_draw
        else:
            pay_share = 0.0

        penalty = overflow * pay_share * penalty_multiplier
        net = gross + bonus - penalty

        payoffs.append(
            AgentPayoff(
                agent_id=agent_id,
                draw_kwh=action.draw_kwh,
                offer_kwh=action.offer_kwh,
                gross_utility=gross,
                share_bonus=bonus,
                penalty=penalty,
                net_payoff=net,
            )
        )

    return RoundResult(
        round_num=round_num,
        payoffs=payoffs,
        total_draw_kwh=total_draw,
        capacity_kwh=capacity_kwh,
        overflow_kwh=overflow,
    )
