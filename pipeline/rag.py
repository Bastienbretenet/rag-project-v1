from dataclasses import dataclass
from typing import Optional

from pipeline.llm import LLMProvider
from pipeline.prompt import build_prompt
from pipeline.retriever import Retriever
from pipeline.vector_store import SearchResult


@dataclass
class RAGAnswer:
    answer: str
    sources: list[SearchResult]


@dataclass
class SourceGroup:
    document_id: str
    title: Optional[str]
    url: Optional[str]
    best_distance: float
    chunks: list[SearchResult]


def group_sources_by_document(sources: list[SearchResult]) -> list[SourceGroup]:
    groups: dict[str, SourceGroup] = {}

    for source in sources:
        group = groups.get(source.document_id)
        if group is None:
            groups[source.document_id] = SourceGroup(
                document_id=source.document_id,
                title=source.title,
                url=source.url,
                best_distance=source.distance,
                chunks=[source],
            )
            continue

        group.chunks.append(source)
        group.best_distance = min(group.best_distance, source.distance)

    ordered_groups = sorted(groups.values(), key=lambda group: group.best_distance)
    for group in ordered_groups:
        group.chunks.sort(key=lambda chunk: chunk.distance)

    return ordered_groups


class RAGPipeline:
    def __init__(self, retriever: Retriever, llm_provider: LLMProvider):
        self._retriever = retriever
        self._llm_provider = llm_provider

    def answer(self, question: str, filters: Optional[dict] = None) -> RAGAnswer:
        chunks = self._retriever.retrieve(question, filters=filters)
        prompt = build_prompt(question, chunks)
        answer = self._llm_provider.generate(prompt)
        return RAGAnswer(answer=answer, sources=chunks)
