from src.ollama import ollama
from src.open_ai import open_ai
from src.strategy_model import ModelStrategy, model_strategy, run_model


@model_strategy("openai")
class OpenAIModel(ModelStrategy):
    def invoke(self, prompt) -> dict:
        response = open_ai.invoke(prompt)
        return response.model_dump()


@model_strategy("ollama")
class OllamaModel(ModelStrategy):
    def invoke(self, prompt) -> dict:
        response = ollama.invoke(prompt)
        return response.model_dump()


result = run_model("openai", "What is the capital of France?")
print(result)


result = run_model("ollama", "What is the capital of Spain?")
print(result)
