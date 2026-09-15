from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    title: str
    date: Optional[str] = None
    url: Optional[str] = None
    domain: Optional[str] = None


class RawDocument(BaseModel):
    id: str
    text: Optional[str] = None
    pdf_path: Optional[str] = None
    metadata: DocumentMetadata


class DocumentSource(ABC):
    @abstractmethod
    def fetch(self, **filters) -> list[RawDocument]:
        ...
