import re
from abc import ABC, abstractmethod
from typing import Optional

import tiktoken
from pydantic import BaseModel

from config import get_settings
from sources.base import RawDocument

TOKEN_ENCODING = "cl100k_base"


class Chunk(BaseModel):
    id: str
    document_id: str
    position: int
    text: str


class Chunker(ABC):
    @abstractmethod
    def split(self, document: RawDocument) -> list[Chunk]:
        ...


class FixedSizeChunker(Chunker):
    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        settings = get_settings()
        self._chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self._chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        self._encoding = tiktoken.get_encoding(TOKEN_ENCODING)

    def split(self, document: RawDocument) -> list[Chunk]:
        tokens = self._encoding.encode(document.text or "")
        step = self._chunk_size - self._chunk_overlap

        chunks = []
        position = 0
        for start in range(0, len(tokens), step):
            token_slice = tokens[start : start + self._chunk_size]
            if not token_slice:
                break

            chunks.append(
                Chunk(
                    id=f"{document.id}-{position}",
                    document_id=document.id,
                    position=position,
                    text=self._encoding.decode(token_slice),
                )
            )
            position += 1

            if start + self._chunk_size >= len(tokens):
                break

        return chunks


class ParagraphChunker(Chunker):
    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        settings = get_settings()
        self._chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self._chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        self._encoding = tiktoken.get_encoding(TOKEN_ENCODING)

    def split(self, document: RawDocument) -> list[Chunk]:
        paragraphs = self._split_into_paragraphs(document.text or "")

        chunks: list[Chunk] = []
        position = 0
        current_tokens: list[int] = []

        for paragraph in paragraphs:
            paragraph_tokens = self._encoding.encode(paragraph)

            if len(paragraph_tokens) > self._chunk_size:
                if current_tokens:
                    chunks.append(self._build_chunk(document.id, position, current_tokens))
                    position += 1
                    current_tokens = []

                for sub_chunk_tokens in self._split_oversized_paragraph(paragraph_tokens):
                    chunks.append(self._build_chunk(document.id, position, sub_chunk_tokens))
                    position += 1
                continue

            if current_tokens and len(current_tokens) + len(paragraph_tokens) > self._chunk_size:
                chunks.append(self._build_chunk(document.id, position, current_tokens))
                position += 1
                current_tokens = self._take_overlap(current_tokens)

            current_tokens.extend(paragraph_tokens)

        if current_tokens:
            chunks.append(self._build_chunk(document.id, position, current_tokens))

        return chunks

    def _split_into_paragraphs(self, text: str) -> list[str]:
        paragraphs = re.split(r"\n\s*\n", text)
        return [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]

    def _take_overlap(self, tokens: list[int]) -> list[int]:
        if self._chunk_overlap <= 0:
            return []
        return tokens[-self._chunk_overlap :]

    def _split_oversized_paragraph(self, tokens: list[int]) -> list[list[int]]:
        step = self._chunk_size - self._chunk_overlap
        return [tokens[start : start + self._chunk_size] for start in range(0, len(tokens), step)]

    def _build_chunk(self, document_id: str, position: int, tokens: list[int]) -> Chunk:
        return Chunk(
            id=f"{document_id}-{position}",
            document_id=document_id,
            position=position,
            text=self._encoding.decode(tokens),
        )
