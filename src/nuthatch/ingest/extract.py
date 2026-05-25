# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Extraction router.

Purpose: detect whether a PDF is digital-text or scanned and dispatch
    to the appropriate extraction backend. Single entry point for the
    Sprint 2 ingest pipeline.

Inputs: a path to a PDF file. Optional explicit strategy override.

Outputs: an `ExtractResult` holding the extracted markdown plus the
    routing decision, page count, and timing.

Assumptions: backend dependencies (`docling`, `easyocr`, the chandra
    CLI, the Granite-Docling VLM) are installed in the active venv.
    GPU detection is via `torch.cuda.is_available()`; absence falls
    back to the CPU strategy.

The routing rules and thresholds derive from the benchmark in
`docs/extraction-benchmarks/mcdonald-kreitman-1991-ocr.md`. Update
both when changing routing behaviour.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

# Threshold below which a PDF is classified as scanned (image-based).
# Source: the McDonald-Kreitman benchmark; the only verifiably image-
# only paper in the test corpus yielded ~106 chars/page, all other
# papers yielded > 1500 chars/page. The threshold sits well above the
# scanned floor and well below any plausible digital-text content.
_SCANNED_TEXT_YIELD_PER_PAGE: float = 200.0


class ExtractionStrategy(StrEnum):
    """The four supported extraction paths."""

    DIGITAL_DOCLING = "digital_docling"
    SCANNED_CHANDRA = "scanned_chandra"
    SCANNED_GRANITE = "scanned_granite"
    SCANNED_EASYOCR = "scanned_easyocr"


@dataclass(frozen=True)
class ExtractResult:
    """The outcome of a single extraction run."""

    text: str
    strategy: ExtractionStrategy
    n_pages: int
    chars_per_page_pre_extraction: float
    wall_clock_seconds: float
    audit: dict[str, str | float | int] = field(default_factory=dict)


def detect_text_yield(pdf_path: Path) -> tuple[float, int]:
    """Return `(chars_per_page, n_pages)` from a pdfminer text extract.

    Used by `pick_strategy` to decide between the digital and scanned
    extraction paths. Cheap (a fraction of a second per typical paper)
    and side-effect-free.
    """
    # Imported lazily so the module loads on hosts without pdfminer.
    from pdfminer.high_level import extract_text
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfparser import PDFParser

    with pdf_path.open("rb") as fh:
        parser = PDFParser(fh)
        doc = PDFDocument(parser)
        n_pages = sum(1 for _ in PDFPage.create_pages(doc))
    text = extract_text(str(pdf_path)) or ""
    chars_per_page = len(text) / max(n_pages, 1)
    return chars_per_page, n_pages


def has_cuda() -> bool:
    """True when a CUDA-capable GPU is available to the active process."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def pick_strategy(
    chars_per_page: float,
    *,
    has_gpu: bool,
    prefer_max_quality: bool = True,
) -> ExtractionStrategy:
    """Decide which backend handles the file.

    Digital-text PDFs route to Docling without OCR. Scanned PDFs route
    to Chandra-OCR-2 when GPU is present and maximum quality is the
    priority, to Granite-Docling when GPU is present but a smaller
    model is preferred, and to Docling+EasyOCR on CPU-only hosts.
    """
    if chars_per_page >= _SCANNED_TEXT_YIELD_PER_PAGE:
        return ExtractionStrategy.DIGITAL_DOCLING
    if has_gpu and prefer_max_quality:
        return ExtractionStrategy.SCANNED_CHANDRA
    if has_gpu:
        return ExtractionStrategy.SCANNED_GRANITE
    return ExtractionStrategy.SCANNED_EASYOCR


def _extract_docling(pdf_path: Path, *, do_ocr: bool) -> str:
    """Docling extraction; OCR off for digital, EasyOCR for scanned."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        EasyOcrOptions,
        PdfPipelineOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    if do_ocr:
        pipe = PdfPipelineOptions(ocr_options=EasyOcrOptions(lang=["en"]), do_ocr=True)
    else:
        pipe = PdfPipelineOptions(do_ocr=False)
    conv = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipe)}
    )
    return conv.convert(str(pdf_path)).document.export_to_markdown()


def _extract_granite(pdf_path: Path) -> str:
    """Docling VLM extraction via the Granite-Docling 258M model."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import VlmPipelineOptions, vlm_model_specs
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline

    vlm_opts = VlmPipelineOptions(vlm_options=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS)
    conv = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline, pipeline_options=vlm_opts
            )
        }
    )
    return conv.convert(str(pdf_path)).document.export_to_markdown()


def _extract_chandra(pdf_path: Path) -> str:
    """Chandra-OCR-2 CLI wrapper. Writes to a temp dir, reads the .md."""
    chandra = shutil.which("chandra")
    if chandra is None:
        raise RuntimeError(
            "chandra CLI not on PATH; install with `sfw uv add chandra-ocr[hf]`"
        )
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        subprocess.run(
            [chandra, str(pdf_path), str(out_dir), "--method", "hf"],
            check=True,
            capture_output=True,
        )
        # Chandra writes to <out>/<stem>/<stem>.md
        stem = pdf_path.stem
        md_path = out_dir / stem / f"{stem}.md"
        return md_path.read_text(encoding="utf-8")


def extract(
    pdf_path: Path,
    *,
    strategy: ExtractionStrategy | None = None,
    prefer_max_quality: bool = True,
) -> ExtractResult:
    """Extract text from a PDF, auto-routing when `strategy` is None."""
    chars_per_page, n_pages = detect_text_yield(pdf_path)
    if strategy is None:
        strategy = pick_strategy(
            chars_per_page,
            has_gpu=has_cuda(),
            prefer_max_quality=prefer_max_quality,
        )

    t0 = time.perf_counter()
    if strategy is ExtractionStrategy.DIGITAL_DOCLING:
        text = _extract_docling(pdf_path, do_ocr=False)
    elif strategy is ExtractionStrategy.SCANNED_EASYOCR:
        text = _extract_docling(pdf_path, do_ocr=True)
    elif strategy is ExtractionStrategy.SCANNED_GRANITE:
        text = _extract_granite(pdf_path)
    elif strategy is ExtractionStrategy.SCANNED_CHANDRA:
        text = _extract_chandra(pdf_path)
    else:
        raise ValueError(f"unknown extraction strategy: {strategy!r}")
    elapsed = time.perf_counter() - t0

    return ExtractResult(
        text=text,
        strategy=strategy,
        n_pages=n_pages,
        chars_per_page_pre_extraction=chars_per_page,
        wall_clock_seconds=elapsed,
        audit={
            "pdf_path": str(pdf_path),
            "threshold_chars_per_page": _SCANNED_TEXT_YIELD_PER_PAGE,
            "extracted_chars": len(text),
        },
    )
