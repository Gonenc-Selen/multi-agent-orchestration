"""Quick Vertex AI connectivity test."""
from __future__ import annotations

from core.llm_client import llm_client
from core.schemas import AgentAction

PROMPT = """You are an energy management agent. Decide how much to draw from the grid.

Current consumption: 3.5 kWh
PV generation: 2.0 kWh
Battery available: 2.0 kWh
Grid capacity: 20.0 kWh

Respond with a JSON object matching the schema."""

print("Testing LLM call with gemini-2.5-flash ...")
action = llm_client.generate(PROMPT, AgentAction)
print(f"  draw_kwh:  {action.draw_kwh}")
print(f"  offer_kwh: {action.offer_kwh}")
print(f"  store_kwh: {action.store_kwh}")
print(f"  reasoning: {action.reasoning[:80]}")
print(f"  tokens — prompt: {llm_client.total_prompt_tokens}, completion: {llm_client.total_completion_tokens}")
print(f"  estimated cost: ${llm_client.estimated_cost_usd:.6f}")
print("SUCCESS")
