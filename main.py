#### your langgraph code
import asyncio, os
from graph import build_graph

async def main():
    graph = build_graph()

    goal = "Plan an outdoor event for 150 people: calculate tables needed (8 per table), check weather in Lahore, and summarize."

    result = await graph.ainvoke({
        "goal":         goal,
        "plan":         [],
        "current_step": 0,
        "results":      [],
        "tools_map":    {},
    })

    print("\n" + "="*60)
    print(" FINAL RESULTS")
    print("="*60)
    for r in result["results"]:
        print(f"\nStep {r['step']}: {r['description']}")
        print(f"  {r['result']}")

if __name__ == "__main__":
    asyncio.run(main())