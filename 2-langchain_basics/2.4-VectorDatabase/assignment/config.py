import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
PDF_PATH = "Principles-of-Data-Science-WEB.pdf"  # Ensure this PDF is in the same directory
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384  # Dimension of "all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "gemma2-9b-it"

# Key to store the main text content within the Qdrant payload
CONTENT_KEY_IN_PAYLOAD = "text_content_for_langchain"
# Define Qdrant collection naming convention
COLLECTION_NAME_PREFIX = "rag_assingment"

# Global client placeholder (will be initialized in main)
qdrant_api_client = None