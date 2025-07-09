from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages

from src.open_ai import open_ai


class State(TypedDict):
    messages: Annotated[list, add_messages]


builder = StateGraph(State)

# model = ChatOpenAI()


def chatbot(state: State):
    answer = open_ai.invoke(state["messages"])
    return {"messages": [answer]}


builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# Add persistence with MemorySaver
graph = builder.compile(checkpointer=MemorySaver())

# Configure thread
thread_1 = {"configurable": {"thread_id": "1"}}

# Run with persistence
result_1 = graph.invoke(
    {"messages": [HumanMessage("hi, my name is Gabriel!")]},  # type: ignore
    thread_1,  # type: ignore
)
print(result_1)
print("\n\n")

result_2 = graph.invoke({"messages": [HumanMessage("what is my name?")]}, thread_1)  # type: ignore
print(result_2)
print("\n\n")

# Get state
# print(graph.get_state(thread_1))  # type: ignore
