from sentence_transformers.cross_encoder import CrossEncoder
from vector_db import retrieve_documents_manually # Import for initial retrieval

def perform_reranking_demonstration(query, collection_name, embeddings_model, client):
    """
    Demonstrates reranking of initially retrieved documents using a Cross-Encoder model.
    """
    reranker_model = None
    try:
        print("  Loading cross-encoder reranker model...")
        reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("  Cross-encoder reranker model loaded.")
    except Exception as e:
        print(f"  Failed to load cross-encoder model: {e}")
        print("  Reranking step skipped. Please ensure you have internet access and 'sentence-transformers' installed.")
        return

    if reranker_model and client is not None:
        k_initial_retrieve_for_reranking = 10

        try:
            client.get_collection(collection_name) # Check if collection exists
        except Exception:
            print(f"  Warning: Collection '{collection_name}' not found. Skipping reranking example.")
            return

        print(f"\\n  Retrieving initial {k_initial_retrieve_for_reranking} documents from '{collection_name}' for reranking...")
        initial_retrieved_docs = retrieve_documents_manually(
            query=query,
            collection_name=collection_name,
            embeddings_model=embeddings_model,
            k=k_initial_retrieve_for_reranking
        )

        if initial_retrieved_docs:
            print(f"  Successfully retrieved {len(initial_retrieved_docs)} documents. Calculating rerank scores...")

            sentence_pairs = [[query, doc.page_content] for doc in initial_retrieved_docs]
            rerank_scores = reranker_model.predict(sentence_pairs)

            docs_with_rerank_scores = list(zip(initial_retrieved_docs, rerank_scores))
            reranked_docs_with_scores = sorted(docs_with_rerank_scores, key=lambda item: item[1], reverse=True)

            print("\\n  Reranked Documents (Top 5):")
            for i, (doc, score) in enumerate(reranked_docs_with_scores[:5]):
                 print(f"    {i+1}. Rerank Score: {score:.4f}, Vector Score: {doc.metadata.get('score', 'N/A'):.4f}, Page: {doc.metadata.get('page', 'N/A')}, Source: {doc.metadata.get('source', 'N/A')}")

            original_order_docs = sorted(initial_retrieved_docs, key=lambda doc: doc.metadata.get('score', -1), reverse=True)
            print("\\n  Original Order Documents (Top 5 by Vector Score):")
            for i, doc in enumerate(original_order_docs[:5]):
                 print(f"    {i+1}. Vector Score: {doc.metadata.get('score', 'N/A'):.4f}, Page: {doc.metadata.get('page', 'N/A')}, Source: {doc.metadata.get('source', 'N/A')}")

            print("\\n  Reranking demonstration complete. Compare the order of documents by Rerank Score vs Vector Score.")
        else:
            print("  No documents retrieved from the initial search for reranking.")