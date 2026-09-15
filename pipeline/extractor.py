import io
import re
from collections import Counter

import pypdf
import requests


def extract_text_from_pdf(pdf_path: str) -> str:
    pdf_file = _open_pdf(pdf_path)
    reader = pypdf.PdfReader(pdf_file)
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages_text)


def _open_pdf(pdf_path: str):
    if pdf_path.startswith("http://") or pdf_path.startswith("https://"):
        response = requests.get(pdf_path, timeout=30)
        response.raise_for_status()
        return io.BytesIO(response.content)
    return pdf_path


def clean_text(text: str) -> str:
    text = _remove_repeated_lines(text)
    text = _remove_ocr_artifacts(text)
    text = _collapse_whitespace(text)
    return text.strip()


def _remove_repeated_lines(text: str) -> str:
    lines = text.split("\n")
    line_counts = Counter(line.strip() for line in lines if line.strip())
    repeated_lines = {line for line, count in line_counts.items() if count > 2}
    return "\n".join(line for line in lines if line.strip() not in repeated_lines)


def _remove_ocr_artifacts(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+", " ", text)


def _collapse_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text
