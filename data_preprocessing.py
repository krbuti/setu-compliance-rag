"""
Parent-child chunking strategy.

Child chunks (256 chars) are stored in ChromaDB for precise vector matching.
Each child carries its parent chunk (1024 chars) in metadata so the retriever
can return richer context than what was matched.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_parent_child_chunks(documents, child_size: int = 256, parent_size: int = 1024):
    child_splitter  = RecursiveCharacterTextSplitter(chunk_size=child_size,  chunk_overlap=50)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_size, chunk_overlap=100)

    child_documents  = []
    parent_documents = []

    for doc in documents:
        content       = doc.page_content
        child_chunks  = child_splitter.split_text(content)
        parent_chunks = parent_splitter.split_text(content)

        for parent_chunk in parent_chunks:
            parent_documents.append({"text": parent_chunk, "metadata": dict(doc.metadata)})

        for i, child in enumerate(child_chunks):
            parent_idx = min(
                int(i * len(parent_chunks) / max(len(child_chunks), 1)),
                len(parent_chunks) - 1,
            )
            child_meta = dict(doc.metadata)
            child_meta["parent_doc"] = parent_chunks[parent_idx]
            child_meta["chunk_id"]   = i
            child_documents.append({"text": child, "metadata": child_meta})

    return child_documents, parent_documents
