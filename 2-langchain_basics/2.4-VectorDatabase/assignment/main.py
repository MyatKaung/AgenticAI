import time

# Import functions and constants from other modules
from config import PDF_PATH, COLLECTION_NAME_PREFIX
from data_loader import extract_pdf_with_sources
from chunking import perform_chunking
import vector_db
from vector_db import (
    initialize_qdrant_client,
    ensure_collections_exist,
    initialize_embeddings,
    prepare_points_for_upsert,
    upsert_chunks_to_qdrant,
    retrieve_documents_manually,
    qdrant_api_client as global_qdrant_client # Import the global client
)
from rag_chain import initialize_llm, run_rag_chain_process
from reranking import perform_reranking_demonstration
from reporter import save_rag_output_to_docx

if __name__ == "__main__":
    # Initialize variables that might be used across steps
    embeddings = None
    llm = None
    chunks = []
    points_to_upsert = []
    collection_names_map = {}
    llm_final_answer = ""
    context_docs_used = []
    user_question = ""

    # --- Step 1 & 2: Data Extraction and Chunking ---
    print(f"--- Starting Data Processing & Qdrant Setup ---")
    print(f"Step 1 & 2: Extracting and chunking data from '{PDF_PATH}'...")

    raw_docs = extract_pdf_with_sources(PDF_PATH)
    raw_docs = [doc for doc in raw_docs if doc.page_content.strip()]
    print(f"  Extracted {len(raw_docs)} raw documents.")

    chunks = perform_chunking(raw_docs)
    if not chunks:
        print("Error: No chunks were created. Cannot proceed.")
    print("Step 1 & 2 Complete.")

    # --- Step 3: Initialize Embeddings ---
    if chunks:
        print(f"\\nStep 3: Initializing embedding model...")
        embeddings = initialize_embeddings()
        if embeddings:
            print("Step 3 Complete: Embedding model initialized.")
        else:
            print("Step 3 Failed: Embedding model not initialized.")
    else:
        print("\\nStep 3 Skipped: No chunks available.")

    # --- Step 4: Initialize Qdrant Client ---
    if embeddings is not None:
        print(f"\\nStep 4: Connecting to Qdrant ...")
        # Initialize the global client from vector_db.py
        vector_db.qdrant_api_client = initialize_qdrant_client()
        if vector_db.qdrant_api_client:
            print("Step 4 Complete: Qdrant client initialized and connected.")
        else:
            print("Step 4 Failed: Qdrant client not initialized.")
    else:
        print("\\nStep 4 Skipped: Embeddings model not initialized.")

    # --- Step 5: Define Index Configurations and Ensure Collections Exist ---
    if vector_db.qdrant_api_client is not None:
        print("\\nStep 5: Ensuring Qdrant collections exist...")
        collection_names_map = ensure_collections_exist(vector_db.qdrant_api_client)
        if collection_names_map:
            print("Step 5 Complete: Collection setup finished.")
        else:
            print("Step 5 Failed: No collections available or created.")
    else:
        print("\\nStep 5 Skipped: Qdrant client not initialized.")

    # --- Step 6: Embed Chunks & Prepare Points ---
    if chunks and embeddings is not None:
        print("\\nStep 6: Embedding chunks and preparing points with corrected payload...")
        points_to_upsert = prepare_points_for_upsert(chunks, embeddings)
        if points_to_upsert:
            print("Step 6 Complete: Points prepared for upsertion.")
        else:
            print("Step 6 Failed: No points prepared for upsertion.")
    else:
        print("\\nStep 6 Skipped: Chunks or embeddings not available.")

    # --- Step 7: Upsert Points into Qdrant (Batched Update) ---
    if points_to_upsert and vector_db.qdrant_api_client is not None and collection_names_map:
        print("\\nStep 7: Upserting/Updating points in Qdrant collections (batched)...")
        upsert_chunks_to_qdrant(vector_db.qdrant_api_client, points_to_upsert, collection_names_map, len(chunks))
        print("Step 7 Complete: Data re-upsertion finished.")
    else:
        print("\\nStep 7 Skipped: Points to upsert, Qdrant client, or collections not available.")

    # --- Step 8: Initialize LLM (Groq) ---
    print("\\nStep 8: Initializing Groq LLM...")
    llm = initialize_llm()
    if llm:
        print("Step 8 Complete: Groq LLM initialized.")
    else:
        print("Step 8 Failed: Groq LLM not initialized.")

    # --- Step 9: Check Retriever Time ---
    if vector_db.qdrant_api_client is not None and embeddings is not None and collection_names_map:
        print("\\nStep 9: Checking Retriever Time (Assignment Item 7)...")
        query_for_timing = "What is the definition of machine learning?"
        k_for_timing = 5

        print(f"--- Measuring Retrieval Time for Query: '{query_for_timing}' (k={k_for_timing}) ---")
        retrieval_times = {}
        collections_for_timing = {k: v for k,v in collection_names_map.items() if v in [c.name for c in vector_db.qdrant_api_client.get_collections().collections]}

        if not collections_for_timing:
             print("  No valid collections available for timing test.")
        else:
            # Warm-up runs
            for _ in range(2):
                 for index_type, col_name in collections_for_timing.items():
                     retrieve_documents_manually(query=query_for_timing, collection_name=col_name, embeddings_model=embeddings, k=k_for_timing)

            # Measure actual time
            for index_type, col_name in collections_for_timing.items():
                print(f"\\n  Querying with '{index_type}' index...")
                start_time = time.time()
                try:
                    retrieved_docs = retrieve_documents_manually(query=query_for_timing, collection_name=col_name, embeddings_model=embeddings, k=k_for_timing)
                    end_time = time.time()
                    duration = end_time - start_time
                    retrieval_times[index_type] = duration
                    print(f"  Retrieved {len(retrieved_docs)} documents in {duration:.4f} seconds.")
                except Exception as e:
                    print(f"  Error querying with '{index_type}': {e}")
                    retrieval_times[index_type] = float('inf')

            print("\\n--- Retrieval Time Summary ---")
            for index_type, duration in retrieval_times.items():
                if duration == float('inf'):
                    print(f"Index: {index_type.upper():<5} - Error during retrieval")
                else:
                    print(f"Index: {index_type.upper():<5} - Time: {duration:.4f} seconds")

            if retrieval_times:
                fastest_index = min(retrieval_times, key=retrieval_times.get)
                if retrieval_times[fastest_index] != float('inf'):
                    print(f"\\nFastest retriever: {fastest_index.upper()} with {retrieval_times[fastest_index]:.4f} seconds.")
                else:
                    print("\\nCould not determine fastest retriever due to errors.")
        print("Step 9 Complete: Retrieval time check finished.")
    else:
        print("\\nStep 9 Skipped: Qdrant client, embeddings, or collections not available.")

    # --- Step 10: Assess Relevance ---
    if vector_db.qdrant_api_client is not None and embeddings is not None and collection_names_map:
        print("\\nStep 10: Assessing Retrieval Relevance (Assignment Item 8)...")
        print("--- Evaluating Relevance of Retrieved Documents ---")
        print("NOTE: This is a qualitative assessment. Manually inspect the output below to judge relevance.")

        queries_for_relevance_check = [
            "Explain the concept of model overfitting in machine learning.",
            "What are common data preprocessing techniques?",
            "Describe the difference between supervised and unsupervised learning.",
            "Find information about data ethics.",
        ]
        k_to_evaluate = 3

        collections_for_relevance = {k: v for k,v in collection_names_map.items() if v in [c.name for c in vector_db.qdrant_api_client.get_collections().collections]}

        if not collections_for_relevance:
             print("  No valid collections available for relevance check.")
        else:
            for query in queries_for_relevance_check:
                print(f"\\n\\nQUERY: {query}")
                for index_type, col_name in collections_for_relevance.items():
                    print(f"\\n  Retriever: {index_type.upper()}")
                    retrieved_docs = retrieve_documents_manually(
                        query=query,
                        collection_name=col_name,
                        embeddings_model=embeddings,
                        k=k_to_evaluate
                    )
                    print(f"  Retrieved {len(retrieved_docs)} documents from {index_type.upper()}. Top {len(retrieved_docs)}:")
                    if retrieved_docs:
                        for i, doc in enumerate(retrieved_docs):
                            print(f"    Doc {i+1} (Page {doc.metadata.get('page', 'N/A')}, Type: {doc.metadata.get('type', 'N/A')}, Source: {doc.metadata.get('source', 'N/A')}, Score: {doc.metadata.get('score', 'N/A'):.4f})")
                            print(f"       Snippet: {doc.page_content[:250]}...")
                    else:
                         print("    No documents retrieved.")

            print("\\n--- Relevance Assessment Complete ---")
            print("Review the output above to compare the relevance of retrieved documents for each index type.")
        print("Step 10 Complete: Relevance assessment finished (manual analysis required).")
    else:
        print("\\nStep 10 Skipped: Qdrant client, embeddings, or collections not available.")

    # --- Step 11: Reranking ---
    if embeddings is not None: # CrossEncoder is imported inside reranking.py
        print("\\nStep 11: Demonstrating Reranking ...")
        print("--- Reranking Example (Cross-Encoder) ---")
        sample_query_for_reranking = "What is supervised learning?"
        collection_for_reranking = collection_names_map.get('hnsw', f"{COLLECTION_NAME_PREFIX}_hnsw")
        perform_reranking_demonstration(sample_query_for_reranking, collection_for_reranking, embeddings, vector_db.qdrant_api_client)
        print("Step 11 Complete: Reranking demonstration finished.")
    else:
        print("\\nStep 11 Skipped: Embeddings not available.")

    # --- Step 12 & 13: Define and Run RAG Chain ---
    if llm is not None and embeddings is not None and vector_db.qdrant_api_client is not None and collection_names_map:
        print("\\nStep 13: Running RAG chain with example query for a specific index type...")
        retrieval_index_type_to_test = 'hnsw'
        collection_to_use = collection_names_map.get(retrieval_index_type_to_test, f"{COLLECTION_NAME_PREFIX}_{retrieval_index_type_to_test}")

        collection_exists = False
        try:
            vector_db.qdrant_api_client.get_collection(collection_to_use)
            collection_exists = True
            print(f"  Using collection '{collection_to_use}' (Index: {retrieval_index_type_to_test.upper()}).")
        except Exception:
            print(f"Warning: Collection '{collection_to_use}' does not exist. Cannot run RAG chain for this type.")

        if collection_exists:
            user_question = "What is the bias-variance trade-off in machine learning, and which pages discuss it?"
            llm_final_answer, context_docs_used = run_rag_chain_process(
                query=user_question,
                collection_name=collection_to_use,
                embeddings_model=embeddings,
                llm_model=llm,
                k_retrieve=10,
                k_context=4
            )

            print("\\n--- Metadata of Context Documents Used for LLM Response ---")
            if context_docs_used:
                 print(f"Context documents used ({len(context_docs_used)}):")
                 for i, doc in enumerate(context_docs_used):
                     print(f"  Doc {i+1}: Page {doc.metadata.get('page', 'N/A')}, Source: {doc.metadata.get('source', 'N/A')}, Score: {doc.metadata.get('score', 'N/A'):.4f}, Type: {doc.metadata.get('type', 'N/A')}")
            else:
                 print("No context documents were used for this response.")
            print("--- End of Context Document Metadata ---")
        else:
            llm_final_answer = f"RAG chain skipped: Collection '{collection_to_use}' not found."
            context_docs_used = []
            print(f"Step 13 Skipped for '{retrieval_index_type_to_test.upper()}': Collection not available.")
    else:
        llm_final_answer = "RAG chain skipped: Dependencies (LLM, embeddings, Qdrant client, collections) not available."
        context_docs_used = []
        print("\\nStep 13 Skipped: Dependencies not met.")
    print("Step 13 Complete.")

    # --- Step 14: Final Output (Save to DOCx) ---
    print("\\nStep 14: Saving LLM output to DOCx...")
    output_filename = "rag_output.docx"
    save_rag_output_to_docx(output_filename, user_question, llm_final_answer, context_docs_used)
    print("Step 14 Complete.")

    print("\\n--- RAG Pipeline Finished ---")