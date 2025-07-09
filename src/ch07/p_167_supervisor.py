from typing import Literal

from langgraph.graph import START, MessagesState, StateGraph
from pydantic import BaseModel

from src.common import show_graph
from src.open_ai import open_ai_4o


class SupervisorDecision(BaseModel):
    next: Literal["researcher", "coder", "FINISH"]


# Initialize model
# model = ChatOpenAI(model="gpt-4", temperature=0)
model = open_ai_4o.with_structured_output(SupervisorDecision)
raw_model = open_ai_4o

# Define available agents
agents = ["researcher", "coder"]

# Define system prompts
system_prompt_part_1 = f"""You are a supervisor tasked with managing a conversation between the  
following workers: {agents}. Given the following user request,  
respond with the worker to act next. Each worker will perform a  
task and respond with their results and status. When finished,  
respond with FINISH."""

system_prompt_part_2 = f"""Given the conversation above, who should act next? 
Or should we FINISH? Select one of: {", ".join(agents)}, FINISH"""


def supervisor(state):
    messages = [
        ("system", system_prompt_part_1),
        *state["messages"],
        ("system", system_prompt_part_2),
    ]
    decision = model.invoke(messages)
    # print(f"Supervisor decision: {decision}")
    # print(f"Supervisor next: {decision.next}")
    response = {
        "messages": state["messages"],
        "next": decision.next,  # type: ignore # properly include the "next" decision
    }
    # print(f"Supervisor response: {response}")
    return response


# Define agent state
class AgentState(MessagesState):
    next: Literal["researcher", "coder", "FINISH"]


# Define agent functions
def researcher(state: AgentState):
    # In a real implementation, this would do research tasks
    print(f"Research next: {state['next']}")
    response = raw_model.invoke(
        [
            {
                "role": "system",
                "content": "You are a research assistant. Analyze the request and provide "
                "relevant information.",
            },
            {"role": "user", "content": state["messages"][0].content},
        ]
    )
    return {"messages": state["messages"] + [response]}  # , "next": state.get("next")}


def coder(state: AgentState):
    # In a real implementation, this would write code
    response = raw_model.invoke(
        [
            {
                "role": "system",
                "content": "You are a coding assistant. Implement the requested functionality.",
            },
            {"role": "user", "content": state["messages"][0].content},
        ]
    )
    return {"messages": state["messages"] + [response]}  # , "next": state.get("next")}


# Build the graph
builder = StateGraph(AgentState)
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher)
builder.add_node("coder", coder)

builder.add_edge(START, "supervisor")
# Route to one of the agents or exit based on the supervisor's decision
builder.add_conditional_edges("supervisor", lambda state: state["next"])
builder.add_edge("researcher", "supervisor")
builder.add_edge("coder", "supervisor")
# builder.add_conditional_edges(
#     "supervisor",
#     lambda state: state["next"],
#     {
#         "researcher": "researcher",
#         "coder": "coder",
#         # Do not specify "FINISH" here; the graph will end if no edge is provided
#     },
# )

graph = builder.compile()

# Example usage
initial_state = {
    "messages": [
        {
            "role": "user",
            "content": "I need help analyzing some data and creating a visualization.",
        }
    ],
    "next": "supervisor",
}

# for output in graph.stream(initial_state):  # type: ignore
#     print(f"\nOutput: {output}")
# for key, value in output.items():
#     print(f"\nType: {key}, Response: {value}")
#     # output_first = output
#     print(f"\nStep decision: {value.get('next', 'N/A')}")
#     if value.get("messages"):
#         print(f"Response: {value['messages'][-1].content[:100]}...")

for output in graph.stream(initial_state):  # type: ignore
    result = output.get("next")
    if not result:
        print("No next step provided. Ending loop.")
        # break

    for key, value in output.items():
        print(f"\nType: {key}, Response: {value}")
        # output_first = output
        print(f"\nStep decision: {value.get('next', 'N/A')}")
        if value.get("messages"):
            print(f"Response: {value['messages'][-1].content[:100]}...")

    # if output.get("next") == "FINISH":
    #     print("✅ Supervisor decided to FINISH. Ending loop.")
    #     break


############
## NOTICE ##
############
# The above code never ends because the supervisor always returns a next step.
# Most probably because the initial_state content is too vague.
#################
show_graph(graph, "p_167_supervisor")
