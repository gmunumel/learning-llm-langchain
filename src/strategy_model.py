from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generator, Type

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

    @abstractmethod
    def batch(self, prompt: list[str]) -> list[Any]:
        """Method to handle batch processing of prompts."""
        pass

    @abstractmethod
    def stream(self, prompt: str) -> Any:
        """Method to handle streaming responses."""
        pass


@dataclass(frozen=True)
class ModelResponse:
    model_name: str
    response: dict | list | str | Any


def run_model_invoke(model_name: str, prompt) -> ModelResponse:
    strategy_cls = model_registry[model_name]
    strategy = strategy_cls()
    response = strategy.invoke(prompt)
    return ModelResponse(model_name=model_name, response=response)


def run_model_batch(model_name: str, prompt) -> ModelResponse:
    strategy_cls = model_registry[model_name]
    strategy = strategy_cls()
    response = strategy.batch(prompt)
    return ModelResponse(model_name=model_name, response=response)


def run_model_stream(model_name: str, prompt) -> Generator[ModelResponse, None, None]:
    strategy_cls = model_registry[model_name]
    strategy = strategy_cls()
    for token in strategy.stream(prompt):
        yield ModelResponse(model_name=model_name, response=token)
