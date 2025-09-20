# packages/providers/layout/table_transformer.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Union
from PIL import Image
import torch
import os

try:
    from transformers import DetrFeatureExtractor, DetrForObjectDetection
    _HAS_HF = True
except Exception:
    _HAS_HF = False


@dataclass
class TATRConfig:
    detect_model: str = "microsoft/table-transformer-detection"           # HF id or local folder
    structure_model: str = "microsoft/table-transformer-structure-recognition"
    conf_thresh: float = 0.6
    device: str = "cpu"                                                   # "cpu" | "cuda" | "mps"


def _resolve_model_path(m: str) -> str:
    # Accept either a HF model id or a local directory path
    p = Path(m)
    return str(p) if p.exists() and p.is_dir() else m


class TableTransformer:
    """
    Thin wrapper around the Table Transformer (DETR) models.
    - detect_tables(img) -> [{"bbox":[x0,y0,x1,y1], "score":...}, ...]
    - structure_blocks(img, bbox) -> [{"bbox":[...], "label":"table row|table column|table column header", "score":...}, ...]
    """
    def __init__(self, cfg: TATRConfig):
        if not _HAS_HF:
            raise RuntimeError("transformers is not installed; cannot use TableTransformer")
        self.cfg = cfg

        det_path = _resolve_model_path(cfg.detect_model)
        str_path = _resolve_model_path(cfg.structure_model)

        # Honor local caches if the user set them
        os.environ.setdefault("HF_HOME", str(Path("models/.cache").absolute()))

        self.det_feat = DetrFeatureExtractor.from_pretrained(det_path)
        self.det_model = DetrForObjectDetection.from_pretrained(det_path).to(cfg.device)

        self.str_feat = DetrFeatureExtractor.from_pretrained(str_path)
        self.str_model = DetrForObjectDetection.from_pretrained(str_path).to(cfg.device)
        self.id2label = self.str_model.config.id2label

    @torch.inference_mode()
    def detect_tables(self, img: Image.Image) -> List[Dict[str, Any]]:
        enc = self.det_feat(images=img, return_tensors="pt")
        enc = {k: v.to(self.cfg.device) for k, v in enc.items()}
        outputs = self.det_model(**enc)
        target_sizes = torch.tensor([img.size[::-1]], device=self.cfg.device)  # (h, w)
        results = self.det_feat.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
        out = []
        for score, box in zip(results["scores"], results["boxes"]):
            sc = float(score)
            if sc < self.cfg.conf_thresh:
                continue
            x0, y0, x1, y1 = [float(v) for v in box]
            out.append({"bbox": [x0, y0, x1, y1], "score": sc})
        return out

    @torch.inference_mode()
    def structure_blocks(self, img: Image.Image, table_bbox) -> List[Dict[str, Any]]:
        x0, y0, x1, y1 = table_bbox
        crop = img.crop((x0, y0, x1, y1))
        enc = self.str_feat(images=crop, return_tensors="pt")
        enc = {k: v.to(self.cfg.device) for k, v in enc.items()}
        outputs = self.str_model(**enc)
        target_sizes = torch.tensor([crop.size[::-1]], device=self.cfg.device)
        results = self.str_feat.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

        blocks = []
        for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
            sc = float(score)
            if sc < self.cfg.conf_thresh:
                continue
            bx0, by0, bx1, by1 = [float(v) for v in box]
            # convert to page coords
            ax0, ay0, ax1, ay1 = x0 + bx0, y0 + by0, x0 + bx1, y0 + by1
            label = self.id2label.get(int(label_id), str(int(label_id)))
            blocks.append({"bbox": [ax0, ay0, ax1, ay1], "label": label, "score": sc})
        return blocks
