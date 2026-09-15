import logging

import streamlit as st

import pipeline
from sources.registry import SOURCE_REGISTRY

st.title("Sources")

st.subheader("Sources enregistrées")
for source_name in SOURCE_REGISTRY:
    st.write(f"- {source_name}")

st.subheader("Lancer une ingestion")
selected_source = st.selectbox("Source", list(SOURCE_REGISTRY))
domain = st.selectbox("Domaine", ["1.info.info-ai"])
date_start = st.date_input("Date de début", value=None)
date_end = st.date_input("Date de fin", value=None)
limit_options = {"5": 5, "25": 25, "50": 50, "100": 100, "Aucune limite": None}
limit_label = st.selectbox("Limite de résultats", list(limit_options), index=1)
limit = limit_options[limit_label]

INGESTION_ENABLED = False

if st.button("Lancer", disabled=True):
    if not INGESTION_ENABLED:
        st.error("Accès refusé : l'ingestion est désactivée.")
        st.stop()

    filters = {"limit": limit}
    if domain:
        filters["domain"] = domain
    if date_start:
        filters["date_start"] = date_start.isoformat()
    if date_end:
        filters["date_end"] = date_end.isoformat()

    log_records: list[str] = []
    progress_placeholder = st.empty()
    progress_bar = st.progress(0)

    def on_progress(processed: int, total: int) -> None:
        progress_placeholder.write(f"{processed}/{total} document(s) traité(s)")
        progress_bar.progress(processed / total)

    class StreamlitLogHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            log_records.append(self.format(record))

    handler = StreamlitLogHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    pipeline_logger = logging.getLogger("pipeline")
    pipeline_logger.addHandler(handler)
    pipeline_logger.setLevel(logging.INFO)

    with st.spinner("Ingestion en cours..."):
        try:
            result = pipeline.run(source=selected_source, filters=filters, on_progress=on_progress)
        finally:
            pipeline_logger.removeHandler(handler)

    st.success(
        f"{result.documents_ingested} document(s) ingéré(s), "
        f"{result.documents_skipped} déjà présent(s), "
        f"{result.chunks_created} chunk(s) créé(s), "
        f"{len(result.errors)} erreur(s)"
    )

    if result.errors:
        st.error("\n".join(result.errors))

    st.subheader("Logs")
    st.code("\n".join(log_records) or "Aucun log.")
