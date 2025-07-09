# pylint: disable=no-name-in-module
from langchain.chains import create_sql_query_chain
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase

from src.open_ai import open_ai

# replace this with the connection details of your db


db = SQLDatabase.from_uri("sqlite:///resources/artifacts/Chinook.db")
print(db.get_usable_table_names())
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# convert question to sql query
write_query = create_sql_query_chain(open_ai, db)

# Execute SQL query
execute_query = QuerySQLDatabaseTool(db=db)

# combined chain = write_query | execute_query
combined_chain = write_query | execute_query

# run the chain
result = combined_chain.invoke({"question": "How many artists are there?"})

print(result)
