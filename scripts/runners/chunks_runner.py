# scripts/runners/chunks_runner.py
# scripts/runners/chunks_runner.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Optional

from packages.core_config.config import load_yaml
from packages.ingest.profiler import profile_document
from packages.ingest.pdf2md_pipeline import PDF2MDConfig, run_pdf_to_markdown
from packages.ingest.chunks import ChunkingConfig, md_to_chunks

def build_pdf2md_cfg(cfg, force_ocr_only: bool, dry_pages: Optional[int]) -> PDF2MDConfig:
    return PDF2MDConfig(
        ocr_enable=bool(cfg.get("ingest.ocr.enable", True)),
        ocr_dpi=int(cfg.get("ingest.ocr.dpi", 300)),
        ocr_lang=str(cfg.get("ingest.ocr.lang", "eng")),
        min_chars_no_ocr=int(cfg.get("ingest.ocr.min_chars_no_ocr", 60)),
        preserve_footnotes=bool(cfg.get("ingest.pdf2md.preserve_footnotes", True)),
        keep_figure_captions=bool(cfg.get("ingest.pdf2md.keep_figure_captions", True)),
        force_ocr_only=force_ocr_only,
        sample_n_pages=int(dry_pages) if dry_pages else None,
        sample_random_seed=123,
        sample_ocr_dpi=int(cfg.get("ingest.ocr.dpi", 300)) if dry_pages else None,
    )

def build_chunk_cfg(cfg) -> ChunkingConfig:
    # Backward-compatible: your dataclass accepts these (new fields have defaults)
    return ChunkingConfig(
        max_chars=int(cfg.get("chunking.max_chars", 1800)),
        overlap=int(cfg.get("chunking.overlap", 100)),
        drop_gibberish=bool(cfg.get("chunking.drop_gibberish", True)),
        drop_toc=bool(cfg.get("chunking.drop_toc", True)),
        min_align_score=int(cfg.get("chunking.min_align_score", 70)),
        attach_heading_to_body=bool(cfg.get("chunking.attach_heading_to_body", True)),
        keep_heading_only=bool(cfg.get("chunking.keep_heading_only", False)),
    )

def run(pdf: Path, providers_yaml="configs/providers.yaml", pipeline_yaml="configs/pipelines/generic_legal.yaml",
        artifacts_root: Optional[Path]=None, dry_pages: Optional[int]=None) -> dict:
    cfg = load_yaml(providers_yaml, pipeline_yaml)
    doc_id = pdf.stem
    art_root = artifacts_root or Path(f"data/artifacts/{doc_id}")
    md_dir, chunks_dir = art_root / "md", art_root / "chunks"
    md_dir.mkdir(parents=True, exist_ok=True); chunks_dir.mkdir(parents=True, exist_ok=True)

    prof = profile_document(pdf)
    force_ocr_only = (prof.mode == "scanned")

    p2m_cfg = build_pdf2md_cfg(cfg, force_ocr_only, dry_pages)
    outs = run_pdf_to_markdown(doc_id=doc_id, pdf_path=pdf, out_dir=md_dir, cfg=p2m_cfg)

    chunk_cfg = build_chunk_cfg(cfg)
    chunks_out = chunks_dir / f"{doc_id}.chunks.jsonl"
    n_chunks = md_to_chunks(
        doc_id=doc_id,
        md_path=outs.markdown_path,
        out_path=chunks_out,
        cfg=chunk_cfg,
        pages_jsonl=outs.pages_jsonl_path,
    )

    return {
        "doc_id": doc_id,
        "artifacts_root": str(art_root),
        "markdown_path": str(outs.markdown_path),
        "pages_jsonl_path": str(outs.pages_jsonl_path) if outs.pages_jsonl_path else None,
        "chunks_path": str(chunks_out),
        "n_chunks": int(n_chunks),
        "profile_mode": prof.mode,
        "pct_texty_pages": getattr(prof, "pct_texty_pages", None),
    }

def main():
    ap = argparse.ArgumentParser(description="PDF → chunks.jsonl")
    ap.add_argument("pdf", type=str)
    ap.add_argument("--providers", default="configs/providers.yaml")
    ap.add_argument("--pipeline", default="configs/pipelines/generic_legal.yaml")
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--dry", type=int, default=None, help="process first N pages only")
    ap.add_argument("--print-sample", type=int, default=3)
    args = ap.parse_args()

    pdf = Path(args.pdf).resolve()
    assert pdf.exists(), f"Missing PDF: {pdf}"

    res = run(
        pdf=pdf,
        providers_yaml=args.providers,
        pipeline_yaml=args.pipeline,
        artifacts_root=Path(args.artifacts) if args.artifacts else None,
        dry_pages=args.dry,
    )

    print(json.dumps(res, ensure_ascii=False, indent=2))

    if args.print_sample > 0:
        print("\n--- sample chunks ---")
        try:
            with open(res["chunks_path"], "r", encoding="utf-8") as f:
                for i, ln in enumerate(f):
                    if i >= args.print_sample: break
                    rec = json.loads(ln)
                    hp = " > ".join(rec["metadata"].get("heading_path") or [])
                    print(f"{i+1}. {rec['id']}  |  {hp}")
                    print(rec["text"][:300] + ("…" if len(rec["text"])>300 else ""))
                    print()
        except Exception as e:
            print(f"(sample failed: {e})")

if __name__ == "__main__":
    main()