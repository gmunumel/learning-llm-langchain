from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

open_ai = ChatOpenAI(model="gpt-3.5-turbo")

open_ai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

open_ai_4o = ChatOpenAI(model="gpt-4o", temperature=0.0)
