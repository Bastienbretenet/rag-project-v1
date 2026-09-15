import streamlit as st

from pipeline.embedder import CachingEmbedder, OpenRouterEmbedder
from pipeline.llm import OpenRouterLLMProvider
from pipeline.rag import RAGPipeline, group_sources_by_document
from pipeline.retriever import Retriever
from pipeline.vector_store import PgVectorStore

st.title("Chat")


def render_sources(sources):
    st.subheader("Sources utilisées")
    for group in group_sources_by_document(sources):
        st.write(f"- **{group.title}** ({group.url}) — distance: {group.best_distance:.4f}")
        with st.expander(f"{len(group.chunks)} extrait(s) utilisé(s)"):
            for chunk in group.chunks:
                st.caption(f"chunk `{chunk.chunk_id}` — distance: {chunk.distance:.4f}")
                st.write(chunk.text)

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if st.button("Effacer la conversation"):
    st.session_state.chat_messages = []
    st.rerun()

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

question = st.chat_input("Poser une question")

if question:
    st.session_state.chat_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    retriever = Retriever(embedder=CachingEmbedder(OpenRouterEmbedder()), vector_store=PgVectorStore())
    rag_pipeline = RAGPipeline(retriever=retriever, llm_provider=OpenRouterLLMProvider())

    with st.chat_message("assistant"):
        with st.spinner("Génération de la réponse..."):
            result = rag_pipeline.answer(question)

        st.write(result.answer)

        render_sources(result.sources)

    st.session_state.chat_messages.append(
        {"role": "assistant", "content": result.answer, "sources": result.sources}
    )
