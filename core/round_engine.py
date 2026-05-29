from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from core.agent import HouseholdAgent
from core.logger import Logger
from core.referee import compute_payoffs
from core.schemas import AgentAction, AgentIntent, AgentState, RoundContext, RoundResult

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_round(self, round_num: int, data_row: pd.Series) -> RoundResult:
        """Run one full round. Returns the RoundResult."""
        mode = self._params.get("communication_mode", "v1")

        if mode == "v2":
            intents = self._collect_intents(round_num, data_row)
            for aid, intent in intents.items():
                self._logger.log_intent(round_num, aid, intent)
            self._intent_log[round_num] = intents
            actions = self._collect_actions(round_num, data_row, neighbor_intents=intents)
        else:
            intents = None
            actions = self._collect_actions(round_num, data_row)

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
        """Intent declarations collected across all rounds (V2 only)."""
        return self._intent_log

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

    def _collect_actions(
        self,
        round_num: int,
        data_row: pd.Series,
        neighbor_intents: dict[str, AgentIntent] | None = None,
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
            action = agent.decide(context, self._params, neighbor_intents=agent_neighbor_intents)
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
                "Round %d | %s: draw=%.2f offer=%.2f store=%.2f "
                "net=%.2f overflow=%.2f",
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
