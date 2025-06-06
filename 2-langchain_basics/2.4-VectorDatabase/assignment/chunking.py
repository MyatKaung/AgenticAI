from langchain.text_splitter import RecursiveCharacterTextSplitter

def perform_chunking(raw_docs):
    """
    Splits raw documents into manageable chunks.
    Text documents are split using RecursiveCharacterTextSplitter,
    while table documents are kept as single chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = []
    for doc in raw_docs:
        if doc.metadata.get("type") == "text":
            splits = text_splitter.split_documents([doc])
            chunks.extend(splits)
        else:
             if doc.page_content.strip():
                 chunks.append(doc)

    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    print(f"Created {len(chunks)} chunks.")
    return chunks