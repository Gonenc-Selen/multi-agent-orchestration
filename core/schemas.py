from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_NEGOTIATION_CATEGORIES = Literal[
    "coordination", "offer_request", "offer_proposal",
    "warning", "agreement", "rejection", "other"
]


class AgentAction(BaseModel):
    draw_kwh: float
    offer_kwh: float
    store_kwh: float
    reasoning: str

    @field_validator("draw_kwh", "offer_kwh", "store_kwh")
    @classmethod
    def must_be_non_negative(cls, v: float, info: object) -> float:
        if v < 0:
            field = getattr(info, "field_name", "field")
            raise ValueError(f"{field} must be >= 0, got {v}")
        return v

    @field_validator("reasoning")
    @classmethod
    def reasoning_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("reasoning must not be empty")
        return v


class AgentIntent(BaseModel):
    intent_draw_kwh: float
    intent_offer_kwh: float
    message: str

    @field_validator("intent_draw_kwh", "intent_offer_kwh")
    @classmethod
    def must_be_non_negative(cls, v: float, info: object) -> float:
        if v < 0:
            field = getattr(info, "field_name", "field")
            raise ValueError(f"{field} must be >= 0, got {v}")
        return v

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be empty")
        return v


class NegotiationOutput(BaseModel):
    """LLM output schema for negotiation — only these 3 fields."""
    to_agent: str
    category: _NEGOTIATION_CATEGORIES = "other"
    message: str

    @field_validator("to_agent", "message")
    @classmethod
    def not_empty(cls, v: str, info: object) -> str:
        if not v.strip():
            field = getattr(info, "field_name", "field")
            raise ValueError(f"{field} must not be empty")
        return v


class NegotiationMessage(BaseModel):
    """Full negotiation record — constructed by Python, never sent to LLM as schema."""
    model_config = ConfigDict(extra="forbid")

    round_num: int
    negotiation_round: int
    from_agent: str = Field(min_length=1)
    to_agent: str = Field(min_length=1)
    category: _NEGOTIATION_CATEGORIES = "other"
    message: str

    @model_validator(mode="after")
    def different_agents(self) -> "NegotiationMessage":
        if self.from_agent == self.to_agent:
            raise ValueError("to_agent must differ from from_agent")
        return self


class RoundContext(BaseModel):
    round_num: int
    agent_id: str
    consumption_kwh: float
    pv_kwh: float
    battery_soc: float
    battery_capacity_kwh: float
    neighbor_actions: dict[str, AgentAction | None]
    memory_summary: str
    season_hint: str


class AgentPayoff(BaseModel):
    agent_id: str
    draw_kwh: float
    offer_kwh: float
    gross_utility: float
    share_bonus: float
    penalty: float
    net_payoff: float


class RoundResult(BaseModel):
    round_num: int
    payoffs: list[AgentPayoff]
    total_draw_kwh: float
    capacity_kwh: float
    overflow_kwh: float


class AgentState(BaseModel):
    agent_id: str
    battery_soc: float
    total_net_payoff: float
    rounds_played: int


class AgentIndividualMetrics(BaseModel):
    net_profit: float
    total_offered_kwh: float
    violation_contribution_kwh: float
    promise_kept_rate: float = 1.0


class RunMetrics(BaseModel):
    total_welfare: float
    capacity_violation_count: int
    capacity_violation_avg_kwh: float
    gini_coefficient: float
    self_sufficiency_ratio: float
    agent_metrics: dict[str, AgentIndividualMetrics]
    communication_mode: str = "v1"
    negotiation_message_count: int = 0


class HouseholdConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    csv: str = Field(min_length=1)
    panel_area_m2: float = Field(gt=0)
    battery_capacity_kwh: float = Field(gt=0)
    persona: str = Field(min_length=1)


class RefereeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit_utility: float = Field(default=1.0, gt=0)
    share_bonus: float = Field(default=0.5, ge=0)
    penalty_multiplier: float = Field(default=3.0, gt=0)


class PVConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    efficiency: float = Field(default=0.18, gt=0, le=1)
    daylight_hours: float = Field(default=10.0, gt=0, le=24)


class PromiseKeepingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tolerance_kwh: float = Field(default=0.5, gt=0)


class ScenarioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    start_date: date
    end_date: date
    season_hint: str = Field(min_length=1)
    grid_capacity_kwh: float = Field(gt=0)
    households: dict[str, HouseholdConfig] = Field(min_length=1)
    referee: RefereeConfig = Field(default_factory=RefereeConfig)
    pv: PVConfig = Field(default_factory=PVConfig)
    communication_mode: Literal["v1", "v2", "v3"] = "v1"
    promise_keeping: PromiseKeepingConfig = Field(default_factory=PromiseKeepingConfig)
    negotiation_rounds: int = Field(default=3, ge=1, le=10)
