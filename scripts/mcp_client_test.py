import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import asyncio
import json
from dotenv import load_dotenv
load_dotenv("../.env")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage


async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],  # runs your server as a subprocess
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # This is the actual MCP step: discover tools FROM the server,
            # rather than binding local Python functions directly
            tools = await load_mcp_tools(session)
            print(f"Discovered {len(tools)} tools via MCP: {[t.name for t in tools]}\n")

            llm = ChatOllama(model="qwen3:8b")
            llm_with_tools = llm.bind_tools(tools)

            question = "Can you find comparable sold homes for a 3-bedroom, 1850 sqft house in Austin? The listing id is L_PINE_101."
            response = await llm_with_tools.ainvoke(question)
            print("Tool calls requested:", response.tool_calls)

            if not response.tool_calls:
                print("No tool call - direct answer:", response.content)
                return

            tool_call = response.tool_calls[0]
            # Find the matching MCP tool object and call it THROUGH the session
            matching_tool = next(t for t in tools if t.name == tool_call["name"])
            result = await matching_tool.ainvoke(tool_call["args"])
            print(f"Tool result (via MCP): {result}")

            messages = [
                HumanMessage(question),
                response,
                ToolMessage(content=str(result), tool_call_id=tool_call["id"]),
            ]
            final_response = await llm_with_tools.ainvoke(messages)
            print(f"\nFinal answer: {final_response.content}")


if __name__ == "__main__":
    asyncio.run(main())