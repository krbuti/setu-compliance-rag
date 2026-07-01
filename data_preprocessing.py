"""
Table-aware parent-child chunking.

Child chunks (≤256 chars) are stored in ChromaDB for precise vector matching.
Each child carries its parent chunk (≤1024 chars) in metadata.

Key improvement over naive splitting:
  Markdown table blocks (lines starting with |) are kept ATOMIC — never split
  across chunks. The nearest preceding section heading is prepended to every
  table chunk so the model understands column semantics even without surrounding
  prose context.

  Example: a leave-entitlements table becomes one chunk like:
      ## Annual Leave Entitlements

      | Grade | Days per Year |
      |-------|--------------|
      | Clerical | 22 |
      | Academic | 30 |

  rather than three disconnected row-chunks.
"""
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

# One or more consecutive markdown table rows (lines that start with |)
_TABLE_RE = re.compile(r'(?:^\|.+\n?)+', re.MULTILINE)

# Any markdown heading (## Heading, # Heading, etc.)
_HEADING_RE = re.compile(r'^#{1,4}\s+.+', re.MULTILINE)


def _last_heading_before(content: str, pos: int) -> str:
    """Return the last markdown heading that appears before `pos` in content."""
    headings = _HEADING_RE.findall(content[:pos])
    return headings[-1].strip() if headings else ""


def _split_with_tables(content: str, splitter: RecursiveCharacterTextSplitter) -> list[str]:
    """
    Split `content` into chunks.

    - Non-table text → split normally with `splitter`
    - Markdown table blocks → kept as a single atomic chunk, prefixed with the
      nearest preceding section heading for context
    """
    chunks: list[str] = []
    cursor = 0

    for match in _TABLE_RE.finditer(content):
        # Text before this table
        text_before = content[cursor:match.start()]
        if text_before.strip():
            chunks.extend(splitter.split_text(text_before))

        # Table block — keep whole, prepend heading for context
        table = match.group(0).rstrip("\n")
        heading = _last_heading_before(content, match.start())
        if heading:
            table = f"{heading}\n\n{table}"
        chunks.append(table)

        cursor = match.end()

    # Remaining text after the last table
    tail = content[cursor:]
    if tail.strip():
        chunks.extend(splitter.split_text(tail))

    return chunks or splitter.split_text(content)


def create_parent_child_chunks(
    documents,
    child_size: int = 256,
    parent_size: int = 1024,
) -> tuple[list[dict], list[dict]]:
    child_splitter  = RecursiveCharacterTextSplitter(chunk_size=child_size,  chunk_overlap=50)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_size, chunk_overlap=100)

    child_documents:  list[dict] = []
    parent_documents: list[dict] = []

    for doc in documents:
        content = doc.page_content

        child_chunks  = _split_with_tables(content, child_splitter)
        parent_chunks = _split_with_tables(content, parent_splitter)

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
