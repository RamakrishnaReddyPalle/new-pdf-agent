# packages/providers/ocr/base.py
from typing import Protocol
from PIL import Image

class OCRProvider(Protocol):
    def image_to_string(self, image: Image.Image, lang: str) -> str: ...
