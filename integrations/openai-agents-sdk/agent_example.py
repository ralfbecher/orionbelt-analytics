"""Example: OrionBelt Analytics agent using OpenAI Agents SDK with MCP.

Prerequisites:
    pip install openai-agents

Start OrionBelt Analytics MCP server first:
    uv run server.py

Then run this script:
    export OPENAI_API_KEY=sk-...
    python agent_example.py
"""

from __future__ import annotations

import asyncio

from agents import Agent, Runner
from agents.mcp import MCPServerStreamableHttp

INSTRUCTIONS = """\
You are a database intelligence assistant powered by OrionBelt Analytics.
You help users analyze database schemas, generate ontologies, and write
safe SQL queries with fan-trap prevention.

## Workflow

1. Call connect_database to establish a connection (postgresql, snowflake,
   dremio, clickhouse, or mysql).
2. Call list_schemas to discover available schemas.
3. Call discover_schema to get the schema structure with relationships.
4. Use get_table_details for deep-dive into specific tables.
5. Call generate_ontology to create RDF/OWL ontology with SQL mappings.
6. Use execute_sql_query to run SQL (set checklist_completed=true) — validation is built-in.
7. Use generate_chart to visualize query results.
8. Use graphrag_query_context for intelligent schema discovery via natural language.

## Rules

- Set checklist_completed=true when calling execute_sql_query.
- Fan-trap warnings must be resolved before executing multi-fact queries.
- Present SQL in code blocks. Explain ontology triples in plain language.
"""

MCP_SERVER_URL = "http://localhost:9000/mcp"


async def main() -> None:
    mcp_server = MCPServerStreamableHttp(url=MCP_SERVER_URL)

    agent = Agent(
        name="OrionBelt Analyst",
        instructions=INSTRUCTIONS,
        model="gpt-4o",
        mcp_servers=[mcp_server],
    )

    # Interactive loop
    print("OrionBelt Analytics Agent (type 'quit' to exit)\n")
    while True:
        user_input = (await asyncio.to_thread(input, "You: ")).strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        result = await Runner.run(agent, user_input)
        print(f"\nAssistant: {result.final_output}\n")


if __name__ == "__main__":
    asyncio.run(main())
