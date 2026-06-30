"""
Phase 9 - Agent with Tool Calling (modern API)

Uses llm.bind_tools() + a simple agentic loop instead of the
deprecated AgentExecutor.
"""
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from models.llm import llm_precise
from tools.tools import ALL_TOOLS


# Bind all tools to the LLM so it can decide which to call.
llm_with_tools = llm_precise.bind_tools(ALL_TOOLS)

# Build a name -> callable lookup for tool execution.
_tool_map = {t.name: t for t in ALL_TOOLS}


def run_agent(query: str, *, max_iterations: int = 10) -> dict:
    """
    Run a tool-calling agent loop.

    Returns:
        dict with keys 'answer' (str) and 'tools_used' (list[str]).
    """
    messages = [HumanMessage(content=query)]
    tools_used: list[str] = []

    for _ in range(max_iterations):
        response: AIMessage = llm_with_tools.invoke(messages)
        messages.append(response)

        # If the model didn't request any tool calls, we're done.
        if not response.tool_calls:
            break

        # Execute each requested tool and feed results back.
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tools_used.append(tool_name)

            tool_fn = _tool_map.get(tool_name)
            if tool_fn is None:
                result = f"Error: unknown tool '{tool_name}'"
            else:
                result = tool_fn.invoke(tool_args)

            messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )

    # The final AI message is the answer.
    answer = messages[-1].content if isinstance(messages[-1], AIMessage) else ""
    return {"answer": answer, "tools_used": tools_used}


if __name__ == "__main__":
    # Quick manual test: uv run python -m chains.agent_chain
    result = run_agent("What is 15 * 8, and what's today's date?")
    print(f"Answer: {result['answer']}")
    print(f"Tools used: {result['tools_used']}")
