from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.common import show_graph
from src.open_ai import open_ai

# Initialize chat model
# model = ChatOpenAI()


# Define state type
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Define prompts
generate_prompt = SystemMessage(
    "You are an essay assistant tasked with writing excellent 3-paragraph essays."
    " Generate the best essay possible for the user's request."
    " If the user provides critique, respond with a revised version of your previous attempts."
)

reflection_prompt = SystemMessage(
    "You are a teacher grading an essay submission. Generate critique and recommendations for the user's submission."
    " Provide detailed recommendations, including requests for length, depth, style, etc."
)


def generate(state: State) -> State:
    answer = open_ai.invoke([generate_prompt] + list(state["messages"]))
    return {"messages": [answer]}


def reflect(state: State) -> State:
    # Invert the messages to get the LLM to reflect on its own output
    cls_map = {AIMessage: HumanMessage, HumanMessage: AIMessage}
    # First message is the original user request. We hold it the same for all nodes
    translated = [reflection_prompt, state["messages"][0]] + [
        cls_map[msg.__class__](content=msg.content) for msg in state["messages"][1:]
    ]
    answer = open_ai.invoke(translated)
    # We treat the output of this as human feedback for the generator
    return {"messages": [HumanMessage(content=answer.content)]}


def should_continue(state: State):
    if len(state["messages"]) > 6:
        # End after 3 iterations, each with 2 messages
        return END
    else:
        return "reflect"


# Build the graph
builder = StateGraph(State)
builder.add_node("generate", generate)
builder.add_node("reflect", reflect)
builder.add_edge(START, "generate")
builder.add_conditional_edges("generate", should_continue)
builder.add_edge("reflect", "generate")

graph = builder.compile()

# Example usage
initial_state = {
    "messages": [
        HumanMessage(
            content="Write an essay about the relevance of 'The Little Prince' today."
        )
    ]
}

# Run the graph
for output in graph.stream(initial_state):  # type: ignore
    message_type = "generate" if "generate" in output else "reflect"
    print("\nNew message:", output[message_type]["messages"][-1].content[:100], "...")

show_graph(graph, "p_156_reflection")
