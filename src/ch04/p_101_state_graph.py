from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph, add_messages

from src.common import show_graph
from src.open_ai import open_ai


class State(TypedDict):
    messages: Annotated[list, add_messages]


builder = StateGraph(State)

# model = ChatOpenAI()


def chatbot(state: State):
    answer = open_ai.invoke(state["messages"])
    return {"messages": [answer]}


# Add the chatbot node
builder.add_node("chatbot", chatbot)

# Add edges
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

# Run the graph
user_input = {"messages": [HumanMessage(content="hi!")]}
for chunk in graph.stream(user_input):  # type: ignore
    print(chunk)

show_graph(graph, "p_101_state_graph")
