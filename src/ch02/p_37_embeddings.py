from src.ollama import ollama_embeddings

embeddings = ollama_embeddings.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!",
    ]
)

print(embeddings)
