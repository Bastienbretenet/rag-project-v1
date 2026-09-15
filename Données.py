import streamlit as st

from pipeline.data_browser import count_chunks, count_documents, list_chunks, list_documents

st.set_page_config(page_title="RAG Project", page_icon="📚")

PAGE_SIZE = 10

st.title("Données")

st.write(
    "Ce projet est un assistant qui répond à tes questions en s'appuyant sur des "
    "publications scientifiques réelles, plutôt que sur ses seules connaissances générales."
)

st.write(
    "Les documents viennent de **HAL** (Hyper Articles en Ligne), une plateforme "
    "française qui regroupe un très large ensemble de publications scientifiques : "
    "articles de recherche, thèses, actes de conférences, etc."
)

st.write(
    "Ici, on se concentre sur les publications du domaine de l'**Intelligence Artificielle**."
)

st.write(
    "Quand tu poses une question, l'assistant recherche les passages les plus pertinents "
    "parmi ces publications, puis rédige une réponse en s'appuyant sur ces sources — "
    "et t'indique toujours les documents utilisés."
)

st.write("**Questions d'exemple :**")
st.write(
    "1. Quels scores d'exposition à l'IA l'auteur calcule-t-il pour les agglomérations "
    "de Caen et de Rouen avec sa méthodologie, et laquelle des deux est la plus exposée ?\n"
    "2. Pourquoi Rouen obtient-elle un score d'exposition à l'IA légèrement supérieur à "
    "celui de Caen, et quelle idée reçue sur les métiers menacés par l'IA ce résultat "
    "vient-il contredire ?\n"
    "3. Quelle performance (en termes d'AUC et de précision) une architecture combinant "
    "CNN et Vision Transformer a-t-elle atteinte pour diagnostiquer la maladie d'Alzheimer "
    "à partir d'IRM structurelles cérébrales ?\n"
    "4. Je travaille sur un projet de diagnostic de la maladie d'Alzheimer par IRM et je "
    "m'intéresse aux approches basées sur les Vision Transformers. Quels travaux existants "
    "utilisent ce type d'architecture pour cette tâche ?"
)

st.subheader("Documents")

total_documents = count_documents()
total_pages = max(1, -(-total_documents // PAGE_SIZE))

if "documents_page" not in st.session_state:
    st.session_state.documents_page = 1

col_prev, col_page, col_next = st.columns([1, 2, 1])
with col_prev:
    if st.button("← Précédent", disabled=st.session_state.documents_page <= 1):
        st.session_state.documents_page -= 1
with col_next:
    if st.button("Suivant →", disabled=st.session_state.documents_page >= total_pages):
        st.session_state.documents_page += 1
with col_page:
    st.write(f"Page {st.session_state.documents_page} / {total_pages} ({total_documents} document(s))")

offset = (st.session_state.documents_page - 1) * PAGE_SIZE
documents = list_documents(offset=offset, limit=PAGE_SIZE)

for document in documents:
    with st.expander(f"{document['title']} — {document['domain']} — {document['date']}"):
        st.write(document["url"])
        st.caption(f"id: {document['id']} — source: {document['source_name']}")

        chunk_total = count_chunks(document["id"])
        st.write(f"{chunk_total} chunk(s)")

        chunk_page_key = f"chunk_page_{document['id']}"
        if chunk_page_key not in st.session_state:
            st.session_state[chunk_page_key] = 1

        chunk_total_pages = max(1, -(-chunk_total // PAGE_SIZE))

        chunk_col_prev, chunk_col_page, chunk_col_next = st.columns([1, 2, 1])
        with chunk_col_prev:
            if st.button(
                "← Précédent",
                key=f"{chunk_page_key}_prev",
                disabled=st.session_state[chunk_page_key] <= 1,
            ):
                st.session_state[chunk_page_key] -= 1
        with chunk_col_next:
            if st.button(
                "Suivant →",
                key=f"{chunk_page_key}_next",
                disabled=st.session_state[chunk_page_key] >= chunk_total_pages,
            ):
                st.session_state[chunk_page_key] += 1
        with chunk_col_page:
            st.write(f"Page {st.session_state[chunk_page_key]} / {chunk_total_pages}")

        chunk_offset = (st.session_state[chunk_page_key] - 1) * PAGE_SIZE
        chunks = list_chunks(document["id"], offset=chunk_offset, limit=PAGE_SIZE)

        for chunk in chunks:
            st.text_area(
                f"Chunk #{chunk['position']}",
                chunk["text"],
                height=100,
                key=f"chunk_text_{document['id']}_{chunk['id']}",
            )
