from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from core.llm_client import LLMClient
from core.memory import MemoryManager
from core.schemas import AgentAction, AgentIntent, RoundContext

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_DEFAULT_REASONING = "Default action: LLM call failed after retries."
_DEFAULT_INTENT_MESSAGE = "Bugünkü tüketimimi karşılamak için şebekeden çekiş yapmayı planlıyorum."
_BATTERY_EFFICIENCY = 0.9


class HouseholdAgent:
    """Single household LLM agent. One agent = one household."""

    def __init__(
        self,
        agent_id: str,
        panel_area_m2: float,
        battery_capacity_kwh: float,
        persona: str,
        llm_client: LLMClient,
        tolerance_kwh: float = 0.5,
    ) -> None:
        self.agent_id = agent_id
        self.panel_area_m2 = panel_area_m2
        self.battery_capacity_kwh = battery_capacity_kwh
        self.battery_soc: float = 0.5
        self.persona = persona
        self.memory = MemoryManager(agent_id=agent_id, tolerance_kwh=tolerance_kwh)
        self._llm = llm_client
        self._jinja_env = Environment(
            loader=FileSystemLoader(str(_PROMPTS_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def declare_intent(
        self, context: RoundContext, scenario_params: dict[str, Any]
    ) -> AgentIntent:
        """Render intent prompt, call LLM, return AgentIntent. Falls back to default on failure."""
        prompt = self._build_intent_prompt(context, scenario_params)
        try:
            return self._llm.generate(prompt, AgentIntent)
        except ValueError as exc:
            logger.error(
                "Agent %s intent LLM failure round %d: %s. Using default intent.",
                self.agent_id,
                context.round_num,
                exc,
            )
            return AgentIntent(
                intent_draw_kwh=context.consumption_kwh,
                intent_offer_kwh=0.0,
                message=_DEFAULT_INTENT_MESSAGE,
            )

    def decide(
        self,
        context: RoundContext,
        scenario_params: dict[str, Any],
        neighbor_intents: dict[str, AgentIntent] | None = None,
    ) -> AgentAction:
        """Render prompts, call LLM, return action. Falls back to default on failure."""
        prompt = self._build_prompt(context, scenario_params, neighbor_intents)
        try:
            action = self._llm.generate(prompt, AgentAction)
        except ValueError as exc:
            logger.error(
                "Agent %s LLM failure round %d: %s. Using default action.",
                self.agent_id,
                context.round_num,
                exc,
            )
            action = AgentAction(
                draw_kwh=context.consumption_kwh,
                offer_kwh=0.0,
                store_kwh=0.0,
                reasoning=_DEFAULT_REASONING,
            )
        return action

    def update_battery_charge(self, store_kwh: float) -> float:
        """Charge battery. Returns actual stored amount (clipped to available capacity)."""
        available = (1.0 - self.battery_soc) * self.battery_capacity_kwh
        actual = min(store_kwh, available)
        if actual < store_kwh:
            logger.warning(
                "Agent %s: store_kwh %.2f clipped to %.2f (battery full).",
                self.agent_id,
                store_kwh,
                actual,
            )
        self.battery_soc += (actual * _BATTERY_EFFICIENCY) / self.battery_capacity_kwh
        self.battery_soc = min(1.0, self.battery_soc)
        return actual

    def discharge_battery(self, kwh_needed: float) -> float:
        """Auto-discharge battery to cover energy need. Returns amount provided."""
        available = self.battery_soc * self.battery_capacity_kwh * _BATTERY_EFFICIENCY
        actual = min(kwh_needed, available)
        self.battery_soc -= actual / (self.battery_capacity_kwh * _BATTERY_EFFICIENCY)
        self.battery_soc = max(0.0, self.battery_soc)
        return actual

    def _build_prompt(
        self,
        context: RoundContext,
        scenario_params: dict[str, Any],
        neighbor_intents: dict[str, AgentIntent] | None = None,
    ) -> str:
        vars_: dict[str, Any] = {
            "agent_id": self.agent_id,
            "persona": self.persona,
            "battery_soc_pct": int(self.battery_soc * 100),
            "battery_capacity_kwh": self.battery_capacity_kwh,
            "battery_available_kwh": round(
                (1.0 - self.battery_soc) * self.battery_capacity_kwh, 2
            ),
            "memory_summary": self.memory.summarize(),
            "neighbor_intents": (
                {aid: i.model_dump() for aid, i in neighbor_intents.items()}
                if neighbor_intents else None
            ),
            **context.model_dump(),
            **scenario_params,
        }
        system = self._jinja_env.get_template("system.j2").render(**vars_)
        decision = self._jinja_env.get_template("decision.j2").render(**vars_)
        return system + "\n\n" + decision

    def _build_intent_prompt(
        self, context: RoundContext, scenario_params: dict[str, Any]
    ) -> str:
        vars_: dict[str, Any] = {
            "agent_id": self.agent_id,
            "persona": self.persona,
            "battery_soc_pct": int(self.battery_soc * 100),
            "battery_capacity_kwh": self.battery_capacity_kwh,
            "battery_available_kwh": round(
                (1.0 - self.battery_soc) * self.battery_capacity_kwh, 2
            ),
            "memory_summary": self.memory.summarize(),
            **context.model_dump(),
            **scenario_params,
        }
        system = self._jinja_env.get_template("system.j2").render(**vars_)
        intent = self._jinja_env.get_template("intent.j2").render(**vars_)
        return system + "\n\n" + intent
