import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dotenv import load_dotenv
load_dotenv("../.env")

import json
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage
from tools import calc_mortgage, get_comps

llm = ChatOllama(model="qwen3:8b")
llm_with_tools = llm.bind_tools([calc_mortgage, get_comps])


def run_agent_query(question):
    response = llm_with_tools.invoke(question)
    print("Tool calls requested:", response.tool_calls)

    if not response.tool_calls:
        print("No tool call - direct answer:", response.content)
        return

    tool_call = response.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    if tool_name == "calc_mortgage":
        result = calc_mortgage(**tool_args)
    elif tool_name == "get_comps":
        result = get_comps(**tool_args)
    else:
        result = None

    print(f"Tool result: {result}")

    messages = [
        HumanMessage(question),
        response,
        ToolMessage(content=json.dumps(result), tool_call_id=tool_call["id"]),
    ]
    final_response = llm_with_tools.invoke(messages)
    print(f"\nFinal answer: {final_response.content}\n")
    print("=" * 60)


run_agent_query("What would my monthly mortgage payment be on a $425,000 home with 20% down at 6.5% interest?")
run_agent_query("Can you find comparable sold homes for a 3-bedroom, 1850 sqft house in Austin? The listing id is L_PINE_101.")