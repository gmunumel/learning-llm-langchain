from langchain_community.document_loaders import TextLoader

loader = TextLoader("resources/test.txt", encoding="utf-8")
docs = loader.load()

print(docs)
