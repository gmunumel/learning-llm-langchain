from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector

from src.open_ai import open_ai, open_ai_embeddings

# See docker command above to launch a postgres instance with pgvector enabled.
CONNECTION = "postgresql+psycopg://vectoruser:vectorpass@localhost:5432/vectordb"

docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
    ),
    Document(
        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
    ),
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={"year": 1995, "genre": "animated"},
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={
            "year": 1979,
            "director": "Andrei Tarkovsky",
            "genre": "thriller",
            "rating": 9.9,
        },
    ),
]

# Create embeddings for the documents
# embeddings_model = OpenAIEmbeddings()

vectorstore = PGVector.from_documents(docs, open_ai_embeddings, connection=CONNECTION)

# Define the fields for the query
fields = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating",
        description="A 1-10 rating for the movie",
        type="float",
    ),
]

description = "Brief summary of a movie"
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
open_ai.temperature = 0.0
retriever = SelfQueryRetriever.from_llm(open_ai, vectorstore, description, fields)

# This example only specifies a filter
print(retriever.invoke("I want to watch a movie rated higher than 8.5"))

print("\n")

# This example specifies multiple filters
print(retriever.invoke("What's a highly rated (above 8.5) science fiction film?"))
