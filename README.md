# AgenticAI Project

This repository contains code and examples for working with Large Language Models (LLMs), Pydantic, and LangChain.

## Project Structure

### 1. Pydantic
Located in `/1-Pydantic/`
- `pydantic.ipynb`: Demonstrates Pydantic usage for data validation including:
  - Basic model creation
  - Field validation
  - Nested models
  - Custom validators
  - Field types and constraints

### 2. LangChain Basics
Located in `/2-langchain_basics/`
- `getting_started_langchain.ipynb`: Introduction to LangChain featuring:
  - Chat models setup (ChatGroq)
  - Prompt templates
  - Output parsers (JSON, XML, String)
  - Chain composition
  - LangSmith integration

- `Assignment.py`: Implementation example showing:
  - Product information extraction
  - Pydantic model integration
  - LLM chain creation

#### 2.1 Data Ingestion
Located in `/2-langchain_basics/2.1-DataIngestion/`
- `dataingestion.ipynb`: Covers various document loaders:
  - Text files
  - PDF documents
  - Web content
  - ArXiv papers
  - Document metadata handling

#### 2.2 Data Transformer
Located in `/2-langchain_basics/2.2-DataTransformer/`
- `2.2.1-Recursivetextsplitter.ipynb`: Text splitting techniques including:
  - RecursiveCharacterTextSplitter
  - CharacterTextSplitter
  - HTMLHeaderTextSplitter
  - JSON splitting
  - Chunk size and overlap configuration

#### 2.3 Embeddings
Located in `/2-langchain_basics/2.3-Embeddings/`
- `2.3.1-embeddings.ipynb`: Basic embedding concepts and implementation
- `2.3.2-hugginface.ipynb`: HuggingFace embeddings integration

#### 2.4 Vector Database
Located in `/2-langchain_basics/2.4-VectorDatabase/`
- `FAISS/`: FAISS vector store implementation
  - `code.ipynb`: FAISS setup and querying
- `Pinecone/`: Pinecone vector database integration
  - `code.ipynb`: Pinecone implementation
- `assignment.ipynb`: RAG implementation tasks
- `check.ipynb`: Vector store validation and testing

## Sample Data Files
- `records.xml`: Example XML data
- `speech.txt`: Sample text for processing
- `syllabus.pdf`: PDF document for demonstration
- `Principles-of-Data-Science-WEB.pdf`: Data science reference material
- `llama2.pdf`: LLama2 model documentation

## Environment Setup
- `.env`: Environment variables configuration
  - QDRANT_API_KEY
  - QDRANT_URL
  - Other API configurations
- `requirements.txt`: Project dependencies

## Getting Started
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables in `.env`
4. Follow the notebooks in order to learn concepts progressively

## Key Topics Covered
- Data validation with Pydantic
- LLM integration with LangChain
- Document processing and text splitting
- Various input/output formats (JSON, XML)
- LangSmith for LLM development
- Embeddings and vector databases
- Retrieval Augmented Generation (RAG)
- FAISS and Pinecone vector stores