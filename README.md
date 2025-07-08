# Learning LLM LangChain

This project is to explore LLM models using LangChain following the book "Learning LangChain"

## Installing Python Virtual Environment

To install a virtual environment for Python execute the following command:

```
uv venv --python 3.12.0
```

Then activate it:

```
source .venv/bin/activate
```

## Install Dependencies

```
uv pip install .
```

## Install and Run Ollama

Install Ollama if not present:

```
curl -fsSL https://ollama.com/install.sh | sh
```

Set the environment variables

```
export OLLAMA_HOST=127.0.0.1 # environment variable to set ollama host
export OLLAMA_PORT=11434 # environment variable to set the ollama port
```

Pull an ollama model:

```
ollama pull llama3.1
```

Run ollama server:

```
ollama serve
```

## Run `pgvector`

`pgvector` is a Postgres database that is able to handle embeddings.

Run it as:

```
docker compose up
```

## Run Files

From the project root run:

```
python -m src.ch01.p_05_llm
```