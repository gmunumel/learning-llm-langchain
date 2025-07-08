from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from src.ollama import ollama
from src.open_ai import open_ai, open_ai_embeddings

# See docker command above to launch a postgres instance with pgvector enabled.
CONNECTION = "postgresql+psycopg://vectoruser:vectorpass@localhost:5432/vectordb"

# Load the document, split it into chunks
raw_documents = TextLoader("resources/test.txt", encoding="utf-8").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

# Create embeddings for the documents
db = PGVector.from_documents(documents, open_ai_embeddings, connection=CONNECTION)

# create retriever to retrieve 2 relevant documents
retriever = db.as_retriever(search_kwargs={"k": 2})

query = "Who are the key figures in the ancient greek history of philosophy?"

# fetch relevant documents
docs = retriever.invoke(query)

print(docs[0].page_content)
print("\n\n")

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context: {context} Question: {question} """
)
open_ai.temperature = 0.0
llm_chain = prompt | open_ai

# answer the question based on relevant documents
result = llm_chain.invoke({"context": docs, "question": query})

print(result)
print("\n\n")

# Run again but this time encapsulate the logic for efficiency

# @chain decorator transforms this function into a LangChain runnable,
# making it compatible with LangChain's chain operations and pipeline

print("Running again but this time encapsulate the logic for efficiency\n")


@chain
def qa(input):
    # fetch relevant documents
    docs = retriever.invoke(input)
    # format prompt
    formatted = prompt.invoke({"context": docs, "question": input})
    # generate answer
    answer = open_ai.invoke(formatted)
    return answer


# run it
result = qa.invoke(query)
print(result.content)
