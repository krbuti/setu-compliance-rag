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


# Splits on any line that starts a markdown heading (used as section boundary)
_HEADING_BOUNDARY_RE = re.compile(r'\n(?=#{1,4} )')


def _prose_chunks(text: str, child_size: int) -> list[str]:
    """Split prose at paragraph breaks; fall back to char splitter for long paragraphs."""
    chunks: list[str] = []
    current = ""
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) > child_size:
            if current:
                chunks.append(current)
                current = ""
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=child_size, chunk_overlap=40,
                separators=["\n", ". ", " ", ""],
            )
            chunks.extend(splitter.split_text(para))
        elif len(current) + len(para) + 2 <= child_size:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if current:
                chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return chunks


def create_semantic_chunks(
    documents,
    child_size: int = 512,
    parent_size: int = 1536,
) -> tuple[list[dict], list[dict]]:
    """Section-aware semantic chunker — splits at heading boundaries, not char counts.

    Each chunk aligns with one policy provision or clause rather than an
    arbitrary character window. Tables remain atomic. The parent stored in
    metadata is the full section text (up to parent_size chars) so the LLM
    always sees a coherent policy section even when the child is small.

    Typical result: ~30% fewer chunks than fixed-size, higher context precision
    because each chunk contains a complete semantic unit.
    """
    child_docs:  list[dict] = []
    parent_docs: list[dict] = []

    for doc in documents:
        content = doc.page_content
        meta    = doc.metadata

        sections = _HEADING_BOUNDARY_RE.split(content)

        for section in sections:
            section = section.strip()
            if len(section) < 40:
                continue

            lines   = section.split("\n")
            heading = lines[0].strip() if _HEADING_RE.match(lines[0]) else ""
            parent_text = section[:parent_size]

            table_matches = list(_TABLE_RE.finditer(section))

            if not table_matches:
                children = (
                    [section] if len(section) <= child_size
                    else _prose_chunks(section, child_size)
                )
            else:
                children: list[str] = []
                cursor = 0
                for tm in table_matches:
                    prose_before = section[cursor:tm.start()].strip()
                    if len(prose_before) >= 40:
                        children += (
                            [prose_before] if len(prose_before) <= child_size
                            else _prose_chunks(prose_before, child_size)
                        )
                    table_text = tm.group(0).rstrip("\n")
                    if heading:
                        table_text = f"{heading}\n\n{table_text}"
                    children.append(table_text)
                    cursor = tm.end()
                tail = section[cursor:].strip()
                if len(tail) >= 40:
                    children += (
                        [tail] if len(tail) <= child_size
                        else _prose_chunks(tail, child_size)
                    )

            for child_text in children:
                if len(child_text.strip()) < 40:
                    continue
                child_meta = {**meta, "parent_doc": parent_text, "heading": heading}
                child_docs.append({"text": child_text.strip(), "metadata": child_meta})
                parent_docs.append({"text": parent_text, "metadata": meta})

    return child_docs, parent_docs


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
