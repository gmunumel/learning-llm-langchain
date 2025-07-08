from langchain_ollama import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama

ollama = ChatOllama(model="llama3.1")

ollama_embeddings = OllamaEmbeddings(model="llama3.1")
