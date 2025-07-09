from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.ollama import ollama
from src.strategy_model import ModelStrategy, model_strategy, run_model


@model_strategy("ollama")
class OllamaModel(ModelStrategy):
    def invoke(self, prompt) -> BaseMessage:
        response = ollama.invoke(prompt)
        return response


system_msg = SystemMessage(
    content="You are a helpful assistant that responds to questions with three exclamation marks."
)
human_msg = HumanMessage(content="What is the capital of Spain?")
result = run_model("ollama", [system_msg, human_msg])
print(result)
