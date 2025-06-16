import fitz  # PyMuPDF
import pdfplumber
import warnings
import os
from langchain.schema import Document

# Suppress specific pdfplumber warning
warnings.filterwarnings("ignore", message="CropBox missing from /Page, defaulting to MediaBox")

def extract_pdf_with_sources(pdf_path):
    """Extracts text and table content from a PDF, retaining source and page info."""
    documents = []
    print(f"Extracting text content from '{os.path.basename(pdf_path)}'...")
    try:
        with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                if text.strip():
                    metadata = {
                        "source": os.path.basename(pdf_path),
                        "page": page_num + 1,
                        "type": "text"
                    }
                    documents.append(Document(page_content=text, metadata=metadata))
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        pass

    print(f"Extracting table content from '{os.path.basename(pdf_path)}'...")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table_num, table_data in enumerate(tables):
                    if table_data:
                        table_content = "\\n".join(["\\t".join(map(str, row)) for row in table_data if row])
                        if table_content.strip():
                            metadata = {
                                "source": os.path.basename(pdf_path),
                                "page": page_num + 1,
                                "table_num": table_num + 1,
                                "type": "table"
                            }
                            documents.append(Document(page_content=f"Table {table_num+1} on page {page_num+1}:\\n{table_content}", metadata=metadata))
    except Exception as e:
        print(f"Error extracting tables from {pdf_path}: {e}")
        pass

    documents = [doc for doc in documents if doc.page_content.strip()]
    print(f"Extracted {len(documents)} raw documents (pages/tables) from '{os.path.basename(pdf_path)}'.")
    return documents

def format_docs_for_context(docs):
    """
    Formats a list of retrieved Documents into a single string suitable for LLM context.
    Includes source information.
    """
    formatted_text = ""
    for i, doc in enumerate(docs):
        source_info_parts = []
        source_info_parts.append(f"Source: {doc.metadata.get('source', 'N/A')}")
        source_info_parts.append(f"Page: {doc.metadata.get('page', 'N/A')}")
        if doc.metadata.get('type') == 'table' and doc.metadata.get('table_num') is not None:
             source_info_parts.append(f"Table: {doc.metadata.get('table_num')}")

        source_info = ", ".join(source_info_parts)

        formatted_text += f"--- Document {i+1} ({source_info}) ---\\n"
        formatted_text += doc.page_content.strip() + "\\n\\n"
    return formatted_text.strip()