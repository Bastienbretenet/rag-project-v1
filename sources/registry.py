from sources.base import DocumentSource
from sources.hal import HALSource

SOURCE_REGISTRY: dict[str, type[DocumentSource]] = {
    "hal": HALSource,
}


def get_source(name: str) -> DocumentSource:
    try:
        source_class = SOURCE_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown source: {name}") from None

    return source_class()
