from langchain_core.prompts import ChatPromptTemplate

from src.ollama import ollama

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ]
)


# combine them with the | operator

chatbot = template | ollama

# use it

response = chatbot.invoke({"question": "Which model providers offer LLMs?"})
print(response.content)

# streaming

# for part in chatbot.stream({"question": "Which model providers offer LLMs?"}):
#     print(part)
