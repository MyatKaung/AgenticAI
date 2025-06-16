import uuid
from qdrant_client import QdrantClient as RawQdrantClient
from qdrant_client.http import models
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import time

# Import constants from config.py
from config import QDRANT_URL, QDRANT_API_KEY, VECTOR_SIZE, CONTENT_KEY_IN_PAYLOAD, COLLECTION_NAME_PREFIX, EMBEDDING_MODEL_NAME

# Global client placeholder (will be initialized in main or passed)
qdrant_api_client = None

def initialize_qdrant_client():
    """Initializes and returns a QdrantClient instance."""
    global qdrant_api_client
    if QDRANT_API_KEY and QDRANT_URL:
        try:
            qdrant_api_client = RawQdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                timeout=60
            )
            print("  Checking Qdrant connection...")
            qdrant_api_client.get_collections()
            print("  Qdrant Connection successful.")
            return qdrant_api_client
        except Exception as e:
            print(f"Error connecting to Qdrant: {e}")
            print("Please check your Qdrant URL, API key, and network connection.")
            return None
    else:
        print("QDRANT_API_KEY or QDRANT_URL not set. Skipping Qdrant client initialization.")
        return None

def ensure_collections_exist(client):
    """Ensures Qdrant collections with specified index configurations exist."""
    if client is None:
        print("Qdrant client not initialized. Cannot ensure collections exist.")
        return {}

    index_configs = {
        "hnsw": models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE, hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100)),
        "flat": models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        "ivf": models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE, quantization_config=models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, quantile=0.99, always_ram=True)))
    }
    collection_names_map = {index_type: f"{COLLECTION_NAME_PREFIX}_{index_type}" for index_type in index_configs.keys()}

    existing_collections = []
    try:
        collections_response = client.get_collections()
        existing_collections = [c.name for c in collections_response.collections]
        print(f"  Currently existing collections: {existing_collections}")
    except Exception as e:
        print(f"  Error listing existing collections: {e}. Proceeding assuming none exist or check failed.")
        existing_collections = []

    for index_type, config in list(index_configs.items()): # Use list to modify while iterating
        collection_name = collection_names_map[index_type]
        if collection_name in existing_collections:
            print(f"  Collection '{collection_name}' already exists. No action needed.")
        else:
            print(f"  Collection '{collection_name}' does not exist. Creating it...")
            try:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=index_configs[index_type]
                )
                print(f"  Collection '{collection_name}' created.")
            except Exception as e:
                 print(f"  Error creating collection '{collection_name}': {e}")
                 print(f"  Warning: Skipping collection '{collection_name}' due to creation error.")
                 del collection_names_map[index_type] # Remove from map if creation fails

    return collection_names_map

def initialize_embeddings():
    """Initializes and returns the HuggingFaceEmbeddings model."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        return embeddings
    except Exception as e:
         print(f"Error initializing embeddings: {e}")
         print("Please ensure 'sentence-transformers' and 'torch' are installed.")
         return None

def prepare_points_for_upsert(chunks, embeddings_model):
    """Embeds chunks and prepares PointStruct objects for Qdrant upsertion."""
    print("  Generating chunk embeddings...")
    chunk_texts_for_embedding = [chunk.page_content for chunk in chunks]
    try:
        chunk_embeddings = embeddings_model.embed_documents(chunk_texts_for_embedding)
        print(f"  Successfully generated {len(chunk_embeddings)} embeddings.")
    except Exception as e:
         print(f"Error generating embeddings: {e}")
         print("Please ensure the embedding model is loaded and working correctly.")
         return []

    points_to_upsert = []
    if chunk_embeddings:
        for i, (chunk, vector) in enumerate(zip(chunks, chunk_embeddings)):
            current_chunk_metadata = {}
            for k, v_meta in chunk.metadata.items():
                if isinstance(v_meta, (str, int, float, bool, list, dict)) or v_meta is None:
                    current_chunk_metadata[k] = v_meta
                else:
                    current_chunk_metadata[k] = str(v_meta)

            payload_for_qdrant = {
                CONTENT_KEY_IN_PAYLOAD: chunk.page_content,
                **current_chunk_metadata
            }
            unique_id = str(uuid.uuid4())

            points_to_upsert.append(
                models.PointStruct(
                    id=unique_id,
                    payload=payload_for_qdrant,
                    vector=vector
                )
            )
    return points_to_upsert

def upsert_chunks_to_qdrant(client, points_to_upsert, collection_names_map, total_chunks):
    """Upserts prepared points into Qdrant collections in batches."""
    if client is None or not points_to_upsert or not collection_names_map:
        print("Skipping upsertion: Qdrant client, points, or collections not available.")
        return

    BATCH_SIZE = 100

    collections_to_upsert = {k: v for k, v in collection_names_map.items() if v in [c.name for c in client.get_collections().collections]}
    if not collections_to_upsert:
        print("  No valid collections available for upsertion.")
        return

    for index_type, collection_name in collections_to_upsert.items():
        print(f"  Upserting/Updating {len(points_to_upsert)} points into '{collection_name}' (Index: {index_type.upper()}) in batches of {BATCH_SIZE}...")

        for i_batch in range(0, len(points_to_upsert), BATCH_SIZE):
            batch_of_points = points_to_upsert[i_batch : i_batch + BATCH_SIZE]
            print(f"  Upserting batch {i_batch // BATCH_SIZE + 1}/{(len(points_to_upsert) + BATCH_SIZE - 1) // BATCH_SIZE} (size: {len(batch_of_points)})")

            try:
                client.upsert(
                    collection_name=collection_name,
                    points=batch_of_points,
                    wait=True
                )
            except Exception as e:
                print(f"    Error upserting batch into '{collection_name}': {e}")
                continue

        print(f"  Finished upserting/updating points into '{collection_name}'.")
        try:
            count_result = client.count(collection_name=collection_name, exact=True)
            print(f"  Collection '{collection_name}' now has {count_result.count} points (should match total chunks: {total_chunks}).")
        except Exception as e:
             print(f"  Error getting count for collection '{collection_name}': {e}")

def retrieve_documents_manually(query: str, collection_name: str, embeddings_model, k: int = 3):
    """
    Performs a vector search using the raw Qdrant client and manually constructs
    LangChain Document objects with correct metadata from the payload.
    """
    global qdrant_api_client # Access the global client
    if qdrant_api_client is None:
        print("Qdrant client is not initialized. Cannot retrieve documents.")
        return []
    try:
        query_vector = embeddings_model.embed_query(query)
        search_result_points = qdrant_api_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=k,
            with_payload=True,
            with_vectors=False
        )

        manually_created_docs = []
        if search_result_points:
            for i, point in enumerate(search_result_points):
                if point.payload and CONTENT_KEY_IN_PAYLOAD in point.payload:
                    doc_content = point.payload[CONTENT_KEY_IN_PAYLOAD]
                    doc_metadata = {k: v for k, v in point.payload.items() if k != CONTENT_KEY_IN_PAYLOAD}
                    doc_metadata["score"] = point.score
                    manually_created_docs.append(
                        Document(page_content=doc_content, metadata=doc_metadata)
                    )
        return manually_created_docs
    except Exception as e:
        print(f"Error during manual search or document construction for '{collection_name}': {e}")
        return []