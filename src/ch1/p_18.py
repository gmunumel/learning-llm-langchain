from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain

from src.ollama import ollama

# the building blocks

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ]
)


# combine them in a function
# @chain decorator adds the same Runnable interface for any function you write


@chain
def chatbot(values):
    prompt = template.invoke(values)
    return ollama.invoke(prompt)


response = chatbot.invoke({"question": "Which model providers offer LLMs?"})
print(response.content)
