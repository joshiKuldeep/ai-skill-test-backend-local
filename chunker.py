"""
Text chunking module.
Designed for logistics documents (BOL, Rate Confirmations, Invoices).

Strategy:
- For structured docs with section markers: chunk by section
- For short docs (< chunk_size): keep as single chunk + create field-level micro-chunks
- For longer docs: line-group based chunking with overlap
- Always creates a "full document" chunk for extraction queries
"""

import re
from dataclasses import dataclass

from config import settings


@dataclass
class Chunk:
    text: str
    index: int
    start_char: int
    end_char: int
    metadata: dict


def _detect_page(text: str) -> list[int]:
    """Extract page numbers from [Page N] markers in text."""
    pages = re.findall(r'\[Page (\d+)\]', text)
    return [int(p) for p in pages] if pages else [1]


def _split_into_sections(text: str) -> list[str]:
    """Split text on section markers (---SECTION: ...--)."""
    parts = re.split(r'---SECTION: .+?---', text)
    headers = re.findall(r'---SECTION: (.+?)---', text)

    sections = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        # Prepend section header if available
        if i > 0 and i - 1 < len(headers):
            part = f"{headers[i - 1]}\n{part}"
        sections.append(part)

    return sections if sections else [text]


def _split_into_line_groups(text: str, group_size: int = 8) -> list[str]:
    """
    Split text into groups of N lines.
    Better than sentence-splitting for structured key-value documents.
    """
    lines = [l for l in text.split("\n") if l.strip()]
    groups = []

    for i in range(0, len(lines), group_size):
        group = "\n".join(lines[i:i + group_size])
        if group.strip():
            groups.append(group.strip())

    return groups


def chunk_text(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
    doc_id: str = "",
) -> list[Chunk]:
    """
    Chunk text into segments optimized for logistics document retrieval.

    Produces multiple chunk types:
    1. Full document chunk (for broad queries and extraction)
    2. Section-based chunks (if section markers present)
    3. Line-group micro-chunks (for precise field-level retrieval)
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    if not text.strip():
        return []

    chunks = []
    chunk_index = 0
    char_offset = 0

    # Clean section markers from display text but use them for splitting
    clean_text = re.sub(r'---SECTION: .+?---\n?', '', text).strip()
    total_words = len(clean_text.split())

    # === CHUNK TYPE 1: Full document chunk ===
    # Always include for broad queries like "summarize this document"
    # and for structured extraction which needs full context
    full_chunk = Chunk(
        text=clean_text[:8000],  # Cap at ~8000 chars
        index=chunk_index,
        start_char=0,
        end_char=len(clean_text[:8000]),
        metadata={
            "doc_id": doc_id,
            "chunk_index": chunk_index,
            "chunk_type": "full_document",
            "pages": _detect_page(text),
            "word_count": min(total_words, len(clean_text[:8000].split())),
        },
    )
    chunks.append(full_chunk)
    chunk_index += 1

    # === CHUNK TYPE 2: Section-based chunks ===
    sections = _split_into_sections(text)

    if len(sections) > 1:
        for section in sections:
            section_clean = re.sub(r'\[Page \d+\]', '', section).strip()
            if not section_clean or len(section_clean.split()) < 3:
                continue

            chunk = Chunk(
                text=section_clean,
                index=chunk_index,
                start_char=char_offset,
                end_char=char_offset + len(section_clean),
                metadata={
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "chunk_type": "section",
                    "pages": _detect_page(section),
                    "word_count": len(section_clean.split()),
                },
            )
            chunks.append(chunk)
            char_offset += len(section_clean)
            chunk_index += 1

    # === CHUNK TYPE 3: Line-group micro-chunks ===
    # These provide precise retrieval for specific field questions
    # "What is the carrier rate?" should match a small chunk with just rate info
    line_groups = _split_into_line_groups(clean_text, group_size=6)

    # Add overlap between groups
    if len(line_groups) > 1:
        overlapped_groups = []
        clean_lines = [l for l in clean_text.split("\n") if l.strip()]

        # Create overlapping windows of 6 lines, stepping by 4
        step = 4
        window = 6
        for i in range(0, len(clean_lines), step):
            group = "\n".join(clean_lines[i:i + window])
            if group.strip() and len(group.split()) >= 3:
                overlapped_groups.append(group.strip())

        line_groups = overlapped_groups

    for group in line_groups:
        # Skip if too similar to full doc chunk (for very short docs)
        if total_words < 30 and len(chunks) > 0:
            continue

        chunk = Chunk(
            text=group,
            index=chunk_index,
            start_char=char_offset,
            end_char=char_offset + len(group),
            metadata={
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "chunk_type": "line_group",
                "pages": _detect_page(group) or _detect_page(text),
                "word_count": len(group.split()),
            },
        )
        chunks.append(chunk)
        char_offset += len(group)
        chunk_index += 1

    return chunks
