# RAG Project

Pipeline RAG (source de documents → chunking → embedding → vector store) avec interface Streamlit.

## Lancer le projet

1. Copier `.env.example` vers `.env` et renseigner les variables (au minimum `DATABASE_URL` et la clé API du provider LLM/embedding choisi).
2. Build + run :

```bash
make rebuild
```

3. Interface Streamlit disponible sur http://localhost:8501, avec quatre pages (menu à gauche) :
   - **Donnée** — parcourt les documents en base (paginé) et les chunks de chaque document (paginé).
   - **Chat** — pose une question, affiche la réponse et les chunks/sources utilisés.
   - **Sources** — liste les sources enregistrées, lance une ingestion (source, domaine, dates) et affiche le statut + logs + progression.

Autres commandes :

```bash
make build   # build l'image
make up      # démarre les containers
make down    # arrête les containers
```

## Ajouter une source

Une source expose des documents normalisés (`RawDocument`) via l'interface `DocumentSource` (`sources/base.py`).

1. Créer `sources/<nom>.py` avec une classe héritant de `DocumentSource` et implémentant `fetch(**filters) -> list[RawDocument]`.
2. Enregistrer la classe dans `sources/registry.py` :

```python
SOURCE_REGISTRY: dict[str, type[DocumentSource]] = {
    "hal": HALSource,
    "ma_source": MaSource,
}
```

3. Utilisation : `get_source("ma_source").fetch(**filtres)`.

## Ajouter une stratégie de chunking

Une stratégie de chunking implémente l'interface `Chunker` (`pipeline/chunker.py`), méthode `split(document) -> list[Chunk]`.

Deux implémentations disponibles :
- `FixedSizeChunker` — coupe à taille fixe (tokens), sans respecter les frontières de paragraphe.
- `ParagraphChunker` (défaut de `IngestionPipeline`) — regroupe les paragraphes jusqu'à `chunk_size` tokens, ne coupe un paragraphe que s'il dépasse `chunk_size` à lui seul. Évite d'isoler un titre de son contenu.

1. Créer une classe héritant de `Chunker` dans `pipeline/chunker.py` (ou un nouveau fichier).
2. Implémenter `split` en respectant le contrat : chaque `Chunk` référence son document parent (`document_id`) et sa position (`position`).
3. Injecter la nouvelle implémentation à la place de `FixedSizeChunker` là où le chunking est utilisé.

## Ajouter une stratégie d'embedding

Une stratégie d'embedding implémente l'interface `Embedder` (`pipeline/embedder.py`), méthode `embed(texts: list[str]) -> list[Vector]`.

1. Créer une classe héritant de `Embedder` dans `pipeline/embedder.py` (ou un nouveau fichier).
2. Injecter la nouvelle implémentation à la place de `OpenRouterEmbedder` là où l'embedding est utilisé.
3. Pour bénéficier du cache (aucun texte ré-embeddé deux fois), l'envelopper avec `CachingEmbedder(mon_embedder)`.

`OpenRouterEmbedder` traite les textes par batch (`batch_size`), avec retry + backoff en cas d'erreur réseau/API. `CachingEmbedder` calcule un hash SHA-256 du texte, vérifie le cache (`embedding_cache`, table pgvector) avant d'appeler l'embedder sous-jacent, et n'y ajoute que les textes manquants.

## Rechercher des chunks proches

`VectorStore` (`pipeline/vector_store.py`) expose `upsert(chunks, vectors)` et `search(query_vector, top_k, filters)`.

`PgVectorStore` stocke les vecteurs dans la table `chunk_vectors` (colonne `vector`, index HNSW `vector_cosine_ops`), liée à `chunks` et `documents`. `search` filtre en SQL sur `documents.domain`/`documents.date` avant de trier par similarité cosinus :

```python
store = PgVectorStore()
results = store.search(query_vector, top_k=5, filters={"domain": "1.info.info-ai"})
```

Filtres supportés : `domain`, `date_start`, `date_end`.

## Lancer une ingestion complète

`IngestionPipeline` (`pipeline/ingestion.py`) enchaîne `source.fetch()` → extraction PDF → `chunker.split()` → `embedder.embed()` → `vector_store.upsert()`, en sautant les documents déjà en base (`DocumentStore.exists`).

```python
import pipeline

result = pipeline.run(source="hal", filters={"domain": "1.info.info-ai"})
```

`result` (`IngestionResult`) contient `documents_fetched`, `documents_ingested`, `documents_skipped`, `chunks_created`, `errors`. Chaque étape logue (module `logging`, logger `pipeline.ingestion`) le nombre de documents/chunks traités et toute erreur par document (une erreur n'interrompt pas les autres documents).

Chaque brique (`chunker`, `embedder`, `vector_store`, `document_store`, `chunk_store`) est injectée dans `IngestionPipeline.__init__` : remplacer une implémentation par défaut se fait en passant un autre objet au constructeur, sans toucher à `run()`.

```python
pipeline_custom = IngestionPipeline(embedder=MonAutreEmbedder())
pipeline_custom.run(source="hal", filters={...})
```

## Interroger en ligne de commande

Un RAG complet est utilisable sans interface, via `cli.py` :

```bash
python cli.py "Quelles sont les avancées récentes en IA ?" --domain 1.info.info-ai
```

Assemblage : `Retriever.retrieve()` (embed la question, interroge `VectorStore`) → `build_prompt()` (`pipeline/prompt.py`, injecte contexte + question) → `LLMProvider.generate()` → `RAGPipeline.answer()` retourne un `RAGAnswer` (`answer` + `sources`, les chunks utilisés).

`LLMProvider` (`pipeline/llm.py`) est une interface interchangeable comme `Embedder`/`VectorStore` : `generate(prompt: str) -> str`. `OpenRouterLLMProvider` est la première implémentation (modèle configuré via `llm_model`). Pour ajouter un autre provider (Anthropic, OpenAI, local...), créer une classe héritant de `LLMProvider` et l'injecter dans `RAGPipeline` à la place de `OpenRouterLLMProvider`.

## Librairies principales

- **streamlit** — interface web du pipeline.
- **pydantic-settings** — configuration centralisée (`config.py`), chargée depuis `.env`, validée au démarrage.
- **requests** — appels à l'API HAL et à l'API OpenRouter (embeddings).
- **pypdf** — extraction du texte des PDF.
- **tiktoken** — tokenization pour le découpage en chunks par taille fixe (encoding `cl100k_base`).
- **psycopg[binary]** — connexion PostgreSQL (stockage `documents`, `chunks`, `embedding_cache`).
- **pgvector** (package Python) — adaptateur du type `vector` pour psycopg.
- **pgvector** (image Docker `pgvector/pgvector:pg16`) — extension PostgreSQL pour le stockage vectoriel.
- **pytest** — tests unitaires.
