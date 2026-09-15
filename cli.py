import argparse

from pipeline.embedder import CachingEmbedder, OpenRouterEmbedder
from pipeline.llm import OpenRouterLLMProvider
from pipeline.rag import RAGPipeline
from pipeline.retriever import Retriever
from pipeline.vector_store import PgVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Interroger le RAG en ligne de commande")
    parser.add_argument("question")
    parser.add_argument("--domain")
    args = parser.parse_args()

    filters = {"domain": args.domain} if args.domain else None

    retriever = Retriever(embedder=CachingEmbedder(OpenRouterEmbedder()), vector_store=PgVectorStore())
    rag_pipeline = RAGPipeline(retriever=retriever, llm_provider=OpenRouterLLMProvider())

    result = rag_pipeline.answer(args.question, filters=filters)

    print(result.answer)
    print("\nSources:")
    for source in result.sources:
        print(f"- {source.title} ({source.url})")


if __name__ == "__main__":
    main()
