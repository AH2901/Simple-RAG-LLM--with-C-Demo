# C# RAG + LLM Demo

A simple RAG and LLM Demo project to answer simple questions from document using C# and Ollama.


## Architecture

```text
Document
   ↓
Text chunks
   ↓
Embedding model
   ↓
In-memory vector search
   ↓
Relevant context
   ↓
LLM
   ↓
Answer
```

## Technologies

* C#
* .NET
* Ollama
* Llama 3.2 1B
* nomic-embed-text
* Cosine similarity
* In-memory vector search

## Setup

Install Ollama and download the models:

```powershell
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

Then open the project in Visual Studio and run the Console App.

## Example

Question:

```text
How many employees does Acme have?
```

The application retrieves the relevant document chunk and sends it to the LLM.

The LLM generates the final answer using the retrieved context.

## Why this project?

This demonstrates the basic RAG pipeline without requiring:

* A paid LLM API
* A vector database
* Azure
* Docker
* A web application

Everything runs locally through Ollama.
