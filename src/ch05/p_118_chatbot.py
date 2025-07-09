from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.common import show_graph
from src.open_ai import open_ai

# model = ChatOpenAI()


class State(TypedDict):
    # Messages have the type "list". The `add_messages`
    # function in the annotation defines how this state should
    # be updated (in this case, it appends new messages to the
    # list, rather than replacing the previous messages)
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    answer = open_ai.invoke(state["messages"])
    return {"messages": [answer]}


builder = StateGraph(State)

builder.add_node("chatbot", chatbot)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

# Example usage

user_input = {"messages": [HumanMessage("hi!")]}
for chunk in graph.stream(user_input):  # type: ignore
    print(chunk)


show_graph(graph, "p_118_chatbot")
