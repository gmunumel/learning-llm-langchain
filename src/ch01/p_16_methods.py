from src.ollama import ollama
from src.strategy_model import (
    ModelStrategy,
    model_strategy,
    run_model,
    run_model_batch,
    run_model_stream,
)


@model_strategy("ollama")
class OllamaModel(ModelStrategy):
    def invoke(self, prompt):
        response = ollama.invoke(prompt)
        return response

    def batch(self, prompts: list[str]) -> list:
        responses = ollama.batch(prompts)  # type: ignore
        return responses

    def stream(self, prompt):
        for token in ollama.stream(prompt):
            yield token


result = run_model("ollama", "Hi there!")
print(result)

result = run_model_batch("ollama", ["Hi there!", "Bye!"])
print(result)

for res_token in run_model_stream("ollama", "Bye!"):
    print(res_token)
