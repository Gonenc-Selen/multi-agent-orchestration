from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from core.agent import HouseholdAgent
from core.logger import Logger
from core.referee import compute_payoffs
from core.schemas import (
    AgentAction,
    AgentIntent,
    AgentState,
    NegotiationMessage,
    RoundContext,
    RoundResult,
)

log = logging.getLogger(__name__)


class RoundEngine:
    """Orchestrates a single game run: builds contexts, collects actions,
    calls referee, updates agent state, and logs everything."""

    def __init__(
        self,
        agents: list[HouseholdAgent],
        scenario_params: dict[str, Any],
        logger: Logger,
    ) -> None:
        self._agents = agents
        self._params = scenario_params
        self._logger = logger
        self._prev_actions: dict[str, AgentAction | None] = {
            a.agent_id: None for a in agents
        }
        self._agent_states: dict[str, AgentState] = {
            a.agent_id: AgentState(
                agent_id=a.agent_id,
                battery_soc=a.battery_soc,
                total_net_payoff=0.0,
                rounds_played=0,
            )
            for a in agents
        }
        self._intent_log: dict[int, dict[str, AgentIntent]] = {}
        self._negotiation_log: dict[int, list[NegotiationMessage]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_round(self, round_num: int, data_row: pd.Series) -> RoundResult:
        """Run one full round (V1/V2/V3 branching). Returns the RoundResult."""
        mode = self._params.get("communication_mode", "v1")

        # --- Phase 1: Intent (V2 and V3) ---
        if mode in ("v2", "v3"):
            intents = self._collect_intents(round_num, data_row)
            for aid, intent in intents.items():
                self._logger.log_intent(round_num, aid, intent)
            self._intent_log[round_num] = intents
        else:
            intents = None

        # --- Phase 2: Negotiation (V3 only) ---
        if mode == "v3":
            neg_msgs = self._collect_negotiations(round_num, data_row, intents)
            self._negotiation_log[round_num] = neg_msgs
        else:
            neg_msgs = None

        # --- Phase 3: Decision ---
        actions = self._collect_actions(
            round_num, data_row,
            neighbor_intents=intents,
            negotiation_msgs=neg_msgs,
        )

        result = self._compute_result(round_num, actions)
        self._update_agents(round_num, data_row, actions, result, round_intents=intents)
        self._logger.log_round_result(round_num, result)
        self._prev_actions = dict(actions)
        return result

    @property
    def agent_states(self) -> list[AgentState]:
        """Current accumulated state for all agents."""
        for agent in self._agents:
            self._agent_states[agent.agent_id].battery_soc = agent.battery_soc
        return list(self._agent_states.values())

    @property
    def intent_log(self) -> dict[int, dict[str, AgentIntent]]:
        return self._intent_log

    @property
    def negotiation_log(self) -> dict[int, list[NegotiationMessage]]:
        return self._negotiation_log

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_intents(
        self, round_num: int, data_row: pd.Series
    ) -> dict[str, AgentIntent]:
        intents: dict[str, AgentIntent] = {}
        for agent in self._agents:
            context = self._build_context(agent, round_num, data_row)
            intent = agent.declare_intent(context, self._params)
            intents[agent.agent_id] = intent
        return intents

    def _collect_negotiations(
        self,
        round_num: int,
        data_row: pd.Series,
        intents: dict[str, AgentIntent] | None,
    ) -> list[NegotiationMessage]:
        assert intents is not None
        neg_rounds = int(self._params.get("negotiation_rounds", 3))
        all_messages: list[NegotiationMessage] = []
        other_ids = {a.agent_id: [x.agent_id for x in self._agents if x.agent_id != a.agent_id]
                     for a in self._agents}

        for neg_round in range(1, neg_rounds + 1):
            for agent in self._agents:
                context = self._build_context(agent, round_num, data_row)
                my_intent = intents[agent.agent_id]
                neighbor_intents = {k: v for k, v in intents.items() if k != agent.agent_id}
                # Only messages sent by this agent or addressed to this agent
                visible = [
                    m for m in all_messages
                    if m.from_agent == agent.agent_id or m.to_agent == agent.agent_id
                ]
                output = agent.send_negotiation_message(
                    context, neg_round, my_intent, neighbor_intents, visible, self._params
                )
                # Guard: to_agent must be a valid neighbor
                valid_targets = other_ids[agent.agent_id]
                target = output.to_agent if output.to_agent in valid_targets else valid_targets[0]
                msg = NegotiationMessage(
                    round_num=round_num,
                    negotiation_round=neg_round,
                    from_agent=agent.agent_id,
                    to_agent=target,
                    category=output.category,
                    message=output.message,
                )
                all_messages.append(msg)
                self._logger.log_negotiation(msg)

        return all_messages

    def _collect_actions(
        self,
        round_num: int,
        data_row: pd.Series,
        neighbor_intents: dict[str, AgentIntent] | None = None,
        negotiation_msgs: list[NegotiationMessage] | None = None,
    ) -> dict[str, AgentAction]:
        actions: dict[str, AgentAction] = {}
        for agent in self._agents:
            context = self._build_context(agent, round_num, data_row)
            agent_neighbor_intents: dict[str, AgentIntent] | None = None
            if neighbor_intents:
                agent_neighbor_intents = {
                    k: v for k, v in neighbor_intents.items()
                    if k != agent.agent_id
                }
            agent_neg_history: list[NegotiationMessage] | None = None
            if negotiation_msgs:
                agent_neg_history = [
                    m for m in negotiation_msgs
                    if m.from_agent == agent.agent_id or m.to_agent == agent.agent_id
                ]
            action = agent.decide(
                context, self._params,
                neighbor_intents=agent_neighbor_intents,
                negotiation_history=agent_neg_history,
            )
            self._logger.log_action(round_num, agent.agent_id, action)
            actions[agent.agent_id] = action
        return actions

    def _compute_result(
        self, round_num: int, actions: dict[str, AgentAction]
    ) -> RoundResult:
        return compute_payoffs(
            actions,
            round_num=round_num,
            capacity_kwh=float(self._params["grid_capacity_kwh"]),
            unit_utility=float(self._params.get("unit_utility", 1.0)),
            share_bonus=float(self._params.get("share_bonus", 0.5)),
            penalty_multiplier=float(self._params.get("penalty_multiplier", 3.0)),
        )

    def _update_agents(
        self,
        round_num: int,
        data_row: pd.Series,
        actions: dict[str, AgentAction],
        result: RoundResult,
        round_intents: dict[str, AgentIntent] | None = None,
    ) -> None:
        payoffs = {p.agent_id: p for p in result.payoffs}

        for agent in self._agents:
            aid = agent.agent_id
            action = actions[aid]
            payoff = payoffs[aid]
            col = aid.lower()
            consumption_kwh = float(data_row[f"{col}_kwh"])
            pv_kwh = float(data_row[f"{col}_pv_kwh"])

            agent.update_battery_charge(action.store_kwh)

            energy_in = pv_kwh + action.draw_kwh
            if energy_in < consumption_kwh:
                agent.discharge_battery(consumption_kwh - energy_in)

            neighbor_actions = {
                other.agent_id: actions[other.agent_id]
                for other in self._agents
                if other.agent_id != aid
            }
            neighbor_intents_for_agent: dict[str, AgentIntent] | None = None
            if round_intents:
                neighbor_intents_for_agent = {
                    k: v for k, v in round_intents.items() if k != aid
                }
            agent.memory.update(
                round_num, action, payoff, neighbor_actions,
                neighbor_intents=neighbor_intents_for_agent,
            )

            state = self._agent_states[aid]
            state.total_net_payoff += payoff.net_payoff
            state.rounds_played += 1
            state.battery_soc = agent.battery_soc

            log.debug(
                "Round %d | %s: draw=%.2f offer=%.2f store=%.2f net=%.2f overflow=%.2f",
                round_num, aid,
                action.draw_kwh, action.offer_kwh, action.store_kwh,
                payoff.net_payoff, result.overflow_kwh,
            )

    def _build_context(
        self, agent: HouseholdAgent, round_num: int, data_row: pd.Series
    ) -> RoundContext:
        col = agent.agent_id.lower()
        return RoundContext(
            round_num=round_num,
            agent_id=agent.agent_id,
            consumption_kwh=float(data_row[f"{col}_kwh"]),
            pv_kwh=float(data_row[f"{col}_pv_kwh"]),
            battery_soc=agent.battery_soc,
            battery_capacity_kwh=agent.battery_capacity_kwh,
            neighbor_actions={
                other.agent_id: self._prev_actions[other.agent_id]
                for other in self._agents
                if other.agent_id != agent.agent_id
            },
            memory_summary=agent.memory.summarize(),
            season_hint=str(self._params.get("season_hint", "")),
        )
