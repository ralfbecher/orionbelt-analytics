"""Example: OrionBelt Analytics agent using LangChain / LangGraph with MCP.

Prerequisites:
    pip install langchain langchain-anthropic langgraph langchain-mcp-adapters

Start OrionBelt Analytics MCP server first:
    uv run server.py

Then run this script:
    export ANTHROPIC_API_KEY=sk-ant-...
    python agent_example.py
"""

from __future__ import annotations

import asyncio

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

SYSTEM_PROMPT = """\
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
- Use GraphRAG tools for natural language schema exploration.
"""

MCP_SERVER_URL = "http://localhost:9000/mcp"


async def main() -> None:
    async with MultiServerMCPClient(
        {
            "orionbelt-analytics": {
                "url": MCP_SERVER_URL,
                "transport": "streamable_http",
            }
        }
    ) as client:
        tools = client.get_tools()

        # Using Anthropic Claude (change to "openai:gpt-4o" for OpenAI)
        agent = create_agent(
            "anthropic:claude-sonnet-4-5",
            tools,
            prompt=SYSTEM_PROMPT,
        )

        # Interactive loop
        print("OrionBelt Analytics Agent (type 'quit' to exit)\n")
        while True:
            user_input = (await asyncio.to_thread(input, "You: ")).strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue

            response = await agent.ainvoke(
                {"messages": [{"role": "user", "content": user_input}]}
            )
            last_message = response["messages"][-1]
            print(f"\nAssistant: {last_message.content}\n")


if __name__ == "__main__":
    asyncio.run(main())
