# packages/providers/ocr/tesseract.py
from __future__ import annotations
from PIL import Image

try:
    import pytesseract
    _HAS_TESS = True
except Exception:
    _HAS_TESS = False

from .base import OCRProvider

class TesseractOCR(OCRProvider):
    def image_to_string(self, image: Image.Image, lang: str = "eng") -> str:
        if not _HAS_TESS:
            raise RuntimeError("pytesseract not installed")
        return pytesseract.image_to_string(image, lang=lang or "eng")
