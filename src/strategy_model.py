from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Type

# Strategy registry
model_registry: Dict[str, Type["ModelStrategy"]] = {}


def model_strategy(name: str):
    """Decorator to register a model strategy class."""

    def decorator(cls: Type["ModelStrategy"]):
        model_registry[name] = cls
        return cls

    return decorator


class ModelStrategy(ABC):
    @abstractmethod
    def invoke(self, prompt) -> Any:
        """Method to invoke the model with a given prompt."""
        pass


@dataclass(frozen=True)
class ModelResponse:
    model_name: str
    response: dict | list | str | Any


def run_model(model_name: str, prompt) -> ModelResponse:
    strategy_cls = model_registry[model_name]
    strategy = strategy_cls()
    response = strategy.invoke(prompt)
    return ModelResponse(model_name=model_name, response=response)
