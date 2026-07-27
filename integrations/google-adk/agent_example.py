"""Example: OrionBelt Analytics agent using Google ADK with MCP.

Prerequisites:
    pip install google-adk

Start OrionBelt Analytics MCP server first:
    uv run server.py

Then run this script:
    export GOOGLE_API_KEY=...
    python agent_example.py
"""

from __future__ import annotations

import asyncio

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool import MCPToolset
from google.genai import types

INSTRUCTIONS = """\
You are a database intelligence assistant powered by OrionBelt Analytics.
You help users analyze database schemas, generate ontologies, and write
safe SQL queries with fan-trap prevention.

Workflow:
1. Call connect_database to establish a connection (postgresql, snowflake,
   dremio, clickhouse, or mysql).
2. Call list_schemas to discover available schemas.
3. Call discover_schema to get the schema structure with relationships.
4. Use get_table_details for deep-dive into specific tables.
5. Call generate_ontology to create RDF/OWL ontology with SQL mappings.
6. Use execute_sql_query to run SQL (set checklist_completed=true) — validation is built-in.
7. Use generate_chart to visualize query results.
8. Use graphrag_query_context for intelligent schema discovery.

Rules:
- Set checklist_completed=true when calling execute_sql_query.
- Fan-trap warnings must be resolved before executing multi-fact queries.
- Present SQL in code blocks. Explain ontology triples in plain language.
"""

MCP_SERVER_URL = "http://localhost:9000/mcp"


async def main() -> None:
    tools, cleanup = await MCPToolset.from_server(
        connection_params={"url": MCP_SERVER_URL},
    )

    agent = Agent(
        name="orionbelt_analyst",
        model="gemini-2.0-flash",
        instruction=INSTRUCTIONS,
        tools=tools,
    )

    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="orionbelt_analyst",
        user_id="user",
    )

    runner = Runner(
        agent=agent,
        app_name="orionbelt_analyst",
        session_service=session_service,
    )

    print("OrionBelt Analytics Agent (type 'quit' to exit)\n")
    try:
        while True:
            user_input = (await asyncio.to_thread(input, "You: ")).strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue

            content = types.Content(
                role="user",
                parts=[types.Part(text=user_input)],
            )

            final_text = ""
            async for event in runner.run_async(
                user_id="user",
                session_id=session.id,
                new_message=content,
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    final_text = event.content.parts[0].text

            print(f"\nAssistant: {final_text}\n")
    finally:
        await cleanup()


if __name__ == "__main__":
    asyncio.run(main())
