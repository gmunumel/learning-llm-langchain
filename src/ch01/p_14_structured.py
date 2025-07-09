from pydantic import BaseModel

from src.ollama import ollama
from src.strategy_model import ModelStrategy, model_strategy, run_model


class AnswerWithJustification(BaseModel):
    """An answer to the user's question along with justification for the answer."""

    answer: str
    """The answer to the user's question"""
    justification: str
    """Justification for the answer"""


@model_strategy("ollama")
class OllamaModel(ModelStrategy):
    def invoke(self, prompt):
        ollama.temperature = 0.0
        structured_llm = ollama.with_structured_output(AnswerWithJustification)
        response = structured_llm.invoke(prompt)
        return response.model_dump()  # type: ignore[return-value]


result = run_model(
    "ollama", "What weighs more, a pound of bricks or a pound of feathers"
)
print(result)
