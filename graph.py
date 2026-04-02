import json, re
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# ─── LLM ──────────────────────────────────────────────────────────────────────
import os
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0,
               api_key=os.environ["GROQ_API_KEY"])

# ─── MCP Tools (math + weather from the provided servers) ─────────────────────
from langchain_mcp_adapters.client import MultiServerMCPClient
import sys

mcp = MultiServerMCPClient({
    "math": {
        "command": sys.executable,
        "args": ["Tools/math_server.py"],
        "transport": "stdio",
    },
    "weather": {
        "url": "http://localhost:8000/mcp",
        "transport": "streamable_http",
    }
})

async def get_mcp_tools(servers: list):
    tools = []
    for server in servers:
        t = await mcp.get_tools(server_name=server)
        tools.extend(t)
    tools_map = {t.name: t for t in tools}
    return tools, tools_map

# ─── State ────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    goal:         str
    plan:         List[dict]
    current_step: int
    results:      List[dict]
    tools_map:    dict          # carries loaded MCP tools through the graph

# ─── Planner Prompt ───────────────────────────────────────────────────────────
PLAN_SYSTEM = """Break the user goal into an ordered JSON list of steps.
Each step MUST follow this EXACT schema:
  {"step": int, "description": str, "tool": str or null, "args": dict or null}

Available tools and their EXACT argument names:
  - calculator(expression: str)         → evaluate a math expression e.g. '150 / 8'
  - get_current_weather(city: str)      → get real weather for a city

Use null for tool/args on synthesis or writing steps.
Return ONLY a valid JSON array. No markdown, no explanation."""

TOOL_ARG_MAP = {
    "calculator":           "expression",
    "get_current_weather":  "city",
}

def safe_args(tool_name: str, raw_args: dict) -> dict:
    expected = TOOL_ARG_MAP.get(tool_name)
    if not expected or expected in raw_args:
        return raw_args
    first_val = next(iter(raw_args.values()), tool_name)
    print(f"  Remapped {raw_args} → {{'{expected}': '{first_val}'}}")
    return {expected: str(first_val)}

# ─── Nodes ────────────────────────────────────────────────────────────────────
async def planner_node(state: AgentState) -> AgentState:
    print(f"\n Goal: {state['goal']}\n")
    # Load MCP tools once here and pass them in state
    _, tools_map = await get_mcp_tools(["math", "weather"])

    resp = llm.invoke([
        SystemMessage(content=PLAN_SYSTEM),
        HumanMessage(content=state["goal"])
    ])
    raw_text  = resp.content if isinstance(resp.content, str) else resp.content[0].get("text", "")
    clean     = re.sub(r"```json|```", "", raw_text).strip()
    plan      = json.loads(clean)

    print(f" Plan ({len(plan)} steps):")
    for s in plan:
        print(f"  Step {s['step']}: {s['description']} | tool={s.get('tool')}")

    return {**state, "plan": plan, "current_step": 0, "results": [], "tools_map": tools_map}


async def executor_node(state: AgentState) -> AgentState:
    idx       = state["current_step"]
    step      = state["plan"][idx]
    tools_map = state["tools_map"]

    print(f"\n  Executing Step {step['step']}: {step['description']}")
    tool_name = step.get("tool")

    if tool_name and tool_name in tools_map:
        corrected = safe_args(tool_name, step.get("args") or {})
        result    = await tools_map[tool_name].ainvoke(corrected)
    else:
        # Synthesis — give LLM context from prior results
        context  = "\n".join([f"Step {r['step']}: {r['result']}" for r in state["results"]])
        response = llm.invoke([
            HumanMessage(content=f"{step['description']}\n\nContext:\n{context}")
        ])
        result = response.content

    print(f"    → {str(result)[:200]}")

    new_results = state["results"] + [{"step": step["step"], "description": step["description"], "result": str(result)}]
    return {**state, "current_step": idx + 1, "results": new_results}


def should_continue(state: AgentState) -> str:
    if state["current_step"] < len(state["plan"]):
        return "executor_node"
    return END


# ─── Build Graph ──────────────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("planner_node",  planner_node)
    g.add_node("executor_node", executor_node)

    g.add_edge(START, "planner_node")
    g.add_edge("planner_node", "executor_node")
    g.add_conditional_edges("executor_node", should_continue)

    return g.compile()