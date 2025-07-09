import uuid

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from src.ollama import ollama_embeddings
from src.open_ai import open_ai_embeddings

# See docker command above to launch a postgres instance with pgvector enabled.
CONNECTION = "postgresql+psycopg://vectoruser:vectorpass@localhost:5432/vectordb"

# Load the document, split it into chunks
raw_documents = TextLoader("resources/artifacts/test.txt", encoding="utf-8").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

# Create embeddings for the documents
db = PGVector.from_documents(documents, open_ai_embeddings, connection=CONNECTION)

results = db.similarity_search("query", k=4)

print(results)

print("Adding documents to the vector store")
ids = [str(uuid.uuid4()), str(uuid.uuid4())]
db.add_documents(
    [
        Document(
            page_content="there are cats in the pond",
            metadata={"location": "pond", "topic": "animals"},
        ),
        Document(
            page_content="ducks are also found in the pond",
            metadata={"location": "pond", "topic": "animals"},
        ),
    ],
    ids=ids,
)

print(
    "Documents added successfully.\n Fetched documents count:", len(db.get_by_ids(ids))
)

print("Deleting document with id", ids[1])
db.delete(ids)

print(
    "Document deleted successfully.\n Fetched documents count:", len(db.get_by_ids(ids))
)
