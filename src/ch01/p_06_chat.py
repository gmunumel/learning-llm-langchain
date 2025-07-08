from langchain_core.messages import BaseMessage, HumanMessage

from src.ollama import ollama
from src.strategy_model import ModelStrategy, model_strategy, run_model_invoke


@model_strategy("ollama")
class OllamaModel(ModelStrategy):
    def invoke(self, prompt) -> BaseMessage:
        response = ollama.invoke(prompt)
        return response


result = run_model_invoke(
    "ollama", [HumanMessage(content="What is the capital of Spain?")]
)
print(result)
