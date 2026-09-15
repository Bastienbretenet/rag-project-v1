from pipeline.vector_store import SearchResult

PROMPT_TEMPLATE = """<context>
{context}
</context>

<instructions>
Réponds à la question en te basant uniquement sur les documents fournis dans <context>.
Si les documents ne contiennent pas l'information nécessaire, dis-le explicitement —
n'invente jamais de réponse.

Chaque extrait de <context> est délimité par une balise <chunk id="...">. Pour citer une
source, reprends exactement cet identifiant entre crochets à la fin de la phrase
concernée, par exemple [hal-05744573_c012]. N'utilise aucun autre format de citation.

Les documents sources peuvent contenir leurs propres références bibliographiques
internes (par exemple [1], [9], [23], ou des noms d'auteurs suivis d'une année).
Ce sont des citations internes au document d'origine, pas des identifiants de chunk :
ignore-les complètement, ne les reproduis jamais dans ta réponse, même si elles
apparaissent à côté de l'information que tu utilises.

Si le contexte fourni décrit la méthode proposée par le document lui-même
(souvent introduite par 'we propose', 'our model', 'this work'), signale-le
clairement comme la contribution du document et non comme un travail antérieur cité.

Réponds en français, de façon directe et concise, sans reformuler la question.
</instructions>

Basé sur les documents ci-dessus, réponds à la question suivante :

<question>
{question}
</question>"""


def build_prompt(question: str, chunks: list[SearchResult]) -> str:
    context = "\n\n".join(f'<chunk id="{chunk.chunk_id}">{chunk.text}</chunk>' for chunk in chunks)
    return PROMPT_TEMPLATE.format(context=context, question=question)
