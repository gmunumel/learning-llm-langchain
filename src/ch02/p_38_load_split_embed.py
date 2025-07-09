from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from src.ollama import ollama_embeddings
from src.open_ai import open_ai_embeddings

# Load the document
loader = TextLoader("resources/artifacts/test.txt", encoding="utf-8")
doc = loader.load()

# Split the document
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(doc)


# Generate embeddings
# ollama_embeddings_result = ollama_embeddings.embed_documents(
#     [chunk.page_content for chunk in chunks]
# )

# print(
#     f"ollama embeddings: {ollama_embeddings_result[:2]}"
# )  # Display first two embeddings

open_ai_embeddings_result = open_ai_embeddings.embed_documents(
    [chunk.page_content for chunk in chunks]
)

print(
    f"open ai embeddings: {open_ai_embeddings_result[:2]}"
)  # Display first two embeddings
