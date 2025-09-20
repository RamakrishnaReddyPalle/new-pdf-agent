# packages/providers/pdf2md/base.py
from typing import Protocol

class PDFToMarkdownProvider(Protocol):
    def to_markdown(self, pdf_path: str) -> str: ...
