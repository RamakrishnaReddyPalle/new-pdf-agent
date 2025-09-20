# Marker/Docling adapters for MD + structure
# packages/providers/pdf2md/marker.py
from __future__ import annotations
from pathlib import Path

try:
    # Docling path (works as a "Marker/Docling" adapter)
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    _HAS_DOCLING = True
except Exception:
    _HAS_DOCLING = False

import fitz  # PyMuPDF fallback

from .base import PDFToMarkdownProvider

class MarkerLikeProvider(PDFToMarkdownProvider):
    """
    Adapter that prefers Docling; falls back to PyMuPDF's markdown/text.
    """
    def __init__(self, preserve_footnotes: bool = True, keep_figure_captions: bool = True):
        self.preserve_footnotes = preserve_footnotes
        self.keep_figure_captions = keep_figure_captions

    def _docling_to_md(self, pdf_path: str) -> str:
        pipeline = PdfPipelineOptions(do_table_structure=True)
        pipeline.table_structure_options.mode = TableFormerMode.ACCURATE
        conv = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)}
        )
        res = conv.convert(str(pdf_path))
        return res.document.export_to_markdown()

    def _pymupdf_to_md(self, pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        parts = []
        for i in range(len(doc)):
            page = doc.load_page(i)
            txt = page.get_text("markdown") or page.get_text("text") or ""
            parts.append(txt.strip())
        return "\n\n---\n\n".join(parts)

    def to_markdown(self, pdf_path: str) -> str:
        if _HAS_DOCLING:
            try:
                return self._docling_to_md(pdf_path)
            except Exception:
                # Fall through gracefully
                pass
        return self._pymupdf_to_md(pdf_path)
