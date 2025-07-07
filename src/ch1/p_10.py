from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate

from src.ollama import ollama
from src.strategy_model import ModelStrategy, model_strategy, run_model_invoke


@model_strategy("ollama")
class OllamaModel(ModelStrategy):
    def invoke(self, prompt) -> BaseMessage:
        response = ollama.invoke(prompt)
        return response


template = PromptTemplate.from_template("""Answer the question based on the context below. 
                                        If the question cannot be answered using the information provided, answer with "I don't know".

Context: {context}

Question: {question}

Answer: """)

user_prompt = template.invoke(
    {
        "context": "The most recent advancements in NLP are being driven by Large Language "
        "Models (LLMs). These models outperform their smaller counterparts and have become "
        "invaluable for developers who are creating applications with NLP capabilities. "
        "Developers can tap into these models through Hugging Face's `transformers` library, "
        "or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` "
        "libraries, respectively.",
        "question": "Which model providers offer LLMs?",
    }
)
result = run_model_invoke("ollama", user_prompt)
print(result)
