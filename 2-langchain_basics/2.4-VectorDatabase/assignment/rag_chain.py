import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Import constants from config.py
from config import GROQ_MODEL_NAME, GROQ_API_KEY
# Import helper functions from data_loader and vector_db
from data_loader import format_docs_for_context
from vector_db import retrieve_documents_manually

# --- Prompt Template Definition ---
prompt_template = ChatPromptTemplate.from_template(
    """You are an AI assistant specialized in data science.
Answer the user's question based *only* on the following context.
If the answer cannot be found in the context, respond with "I cannot answer the question based on the provided information."
Include the source page number and source file name for each piece of information you use from the context. Quote directly from the source when providing specific facts or definitions.

Context:
{context}

Question: {question}

Answer:"""
)

def initialize_llm():
    """Initializes and returns the ChatGroq LLM instance."""
    if not GROQ_API_KEY:
        print("WARNING: GROQ_API_KEY is not set.")
        return None
    try:
        llm = ChatGroq(model=GROQ_MODEL_NAME, temperature=0.1, groq_api_key=GROQ_API_KEY)
        print("  Testing LLM connection...")
        test_response = llm.invoke("Hi, how are you?", max_tokens=10)
        print(f"  LLM test response: {test_response.content}")
        print(f"Groq LLM '{GROQ_MODEL_NAME}' initialized.")
        return llm
    except Exception as e:
        print(f"Error initializing Groq LLM: {e}")
        print("Please ensure 'langchain-groq' is installed and GROQ_API_KEY is set correctly.")
        return None

def run_rag_chain_process(query: str, collection_name: str, embeddings_model, llm_model, k_retrieve: int = 10, k_context: int = 4):
    """
    Runs the RAG chain process for a single query:
    1. Retrieves docs manually from Qdrant.
    2. Selects top N docs for context
    3. Formats context string.
    4. Creates prompt string.
    5. Invokes LLM.
    """
    if llm_model is None:
        return "LLM is not initialized.", []

    print(f"\\n--- Running RAG Chain for Query: '{query}' on '{collection_name}' ---")

    print(f"  Retrieving top {k_retrieve} documents...")
    retrieved_docs = retrieve_documents_manually(
        query=query,
        collection_name=collection_name,
        embeddings_model=embeddings_model,
        k=k_retrieve
    )

    if not retrieved_docs:
        print("  Could not retrieve relevant documents.")
        return "Could not retrieve relevant documents for the query.", []

    docs_for_context_selection = retrieved_docs

    final_docs_for_context = docs_for_context_selection[:k_context]
    print(f"  Using top {len(final_docs_for_context)} documents for context.")

    if not final_docs_for_context:
        print("  No documents selected for context.")
        return "No documents selected for context.", []

    context_string = format_docs_for_context(final_docs_for_context)

    final_prompt_string = prompt_template.format(context=context_string, question=query)

    print("  Generating LLM Response...")
    start_time = time.time()
    try:
        llm_response_obj = llm_model.invoke(final_prompt_string)
        llm_response = llm_response_obj.content
        end_time = time.time()
        print(f"  LLM Response generated in {end_time - start_time:.2f} seconds.")

        print("\\nLLM Response:")
        print(llm_response)

        return llm_response, final_docs_for_context

    except Exception as e:
        print(f"  Error invoking LLM: {e}")
        return "Error generating response from LLM.", []