"""
PDF Processing Handler for extracting figures and tables from academic papers.

Uses pymupdf4llm for robust, LLM-optimised extraction of images and tables
from PDF documents.  pymupdf4llm handles multi-column layouts, colorspace
quirks and table detection out of the box.
"""

import hashlib
import logging
import mmap
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Dict, List, Optional, Set, Tuple
import fitz  # PyMuPDF

from pymupdf4llm.helpers.pymupdf_rag import to_markdown as _to_markdown

from PIL import Image
import io
import re

from models.schemas.nodes import Figure, Table, Paper
from clients.huggingface.CLIPClient import CLIPClient
from clients.huggingface.SciBERTClient import SciBERTClient
from clients.vector.MilvusClient import MilvusClient


# ─── Pre-compiled regexes (module-level for zero per-call cost) ──────
_RE_IMAGE_TAG = re.compile(
    r'!\[([^\]]*)\]\(([^)]+\.(?:png|jpg|jpeg|webp|gif))\)',
    re.IGNORECASE,
)
_RE_FIG_CAPTION = re.compile(
    r'((?:fig(?:ure|\.)?\s*\.?\s*(?:S?\d+[a-z]?(?:\([a-z]\))?))[.:—\-]?\s*.+?)'
    r'(?:\n\n|\n(?=#{1,3}\s)|\n(?=(?:fig(?:ure|\.)?|tab(?:le|\.)?|\|)\s*\.?\s*\d)|$)',
    re.IGNORECASE | re.DOTALL,
)
_RE_FIG_BOLD_CAPTION = re.compile(
    r'(\*{1,2}(?:fig(?:ure|\.)?\s*\.?\s*S?\d+[a-z]?)\*{1,2}[.:—\-]?\s*.+?)'
    r'(?:\n\n|\n(?=#{1,3}\s)|$)',
    re.IGNORECASE | re.DOTALL,
)
_RE_HAS_FIGURE_WORD = re.compile(r'\bfig(?:ure)?\b', re.IGNORECASE)
_RE_HAS_TABLE_WORD = re.compile(r'\btab(?:le)?\b', re.IGNORECASE)
_RE_SEP_CELL = re.compile(r'^:?\s*-{3,}\s*:?$')
_RE_EXTRA_SEP_ROW = re.compile(r'^[\s|:-]+$')

# ─── Pre-compiled caption patterns (were re.compile'd per-call!) ─────
# Figure caption: "Figure 1: …" / "Fig. 2. …" / "Figure S3a …"
_RE_FIG_CAP_STD = re.compile(
    r'((?:fig(?:ure|\.)?\s*\.?\s*(?:S?\d+[a-z]?(?:\([a-z]\))?))[.:—\-]?\s*.+?)'
    r'(?:\n\n|\n(?=#{1,3}\s)|\n(?=(?:fig(?:ure|\.)?|tab(?:le|\.)?|\|)\s*\.?\s*\d)|$)',
    re.IGNORECASE | re.DOTALL,
)
# Bold/italic figure caption: "**Figure 1.** description"
_RE_FIG_CAP_BOLD = re.compile(
    r'(\*{1,2}(?:fig(?:ure|\.)?\s*\.?\s*S?\d+[a-z]?)\*{1,2}[.:—\-]?\s*.+?)'
    r'(?:\n\n|\n(?=#{1,3}\s)|$)',
    re.IGNORECASE | re.DOTALL,
)
# Table caption: "Table 1: …" / "Tab. 2. …"
_RE_TBL_CAP_STD = re.compile(
    r'((?:tab(?:le|\.)?\s*\.?\s*(?:S?\d+[a-z]?))[.:—\-]?\s*.+?)'
    r'(?:\n\n|\n(?=\|)|\n(?=#{1,3}\s)|\n(?=(?:fig(?:ure|\.)?|tab(?:le|\.)?)\.?\s*\d)|$)',
    re.IGNORECASE | re.DOTALL,
)
# Bold/italic table caption
_RE_TBL_CAP_BOLD = re.compile(
    r'(\*{1,2}(?:tab(?:le|\.)?\s*\.?\s*S?\d+[a-z]?)\*{1,2}[.:—\-]?\s*.+?)'
    r'(?:\n\n|\n(?=\|)|$)',
    re.IGNORECASE | re.DOTALL,
)
# Caption in raw page text (native fallback)
_RE_PAGE_CAP_FIG = re.compile(
    r'((?:fig(?:ure|\.)?\s*\.?\s*(?:S?\d+[a-z]?(?:\([a-z]\))?))[.:—\-]?\s*.+?)'
    r'(?:\n\n|\n(?=(?:fig(?:ure|\.)?|tab(?:le|\.)?)\.?\s*\d)|$)',
    re.IGNORECASE | re.DOTALL,
)


class PDFProcessingHandler:
    """Handler for processing PDFs and extracting visual content."""

    MAX_FIGURES_PER_PAPER = 10   # cap extracted figures per PDF
    MAX_TABLES_PER_PAPER  = 10   # cap extracted tables per PDF
    MAX_PAGES_TO_PROCESS  = 30   # only scan the first N pages for figures/tables
    EXTRACTION_TIMEOUT    = 120  # seconds – skip paper if extraction exceeds this
    DOWNLOAD_TIMEOUT      = 60   # seconds – skip paper if PDF download exceeds this
    MARKDOWN_TIMEOUT      = 10   # seconds – hard cap per _to_markdown() attempt

    def __init__(self, clip_client: Optional[CLIPClient] = None, scibert_client: Optional[SciBERTClient] = None,
                 milvus_client: Optional[MilvusClient] = None):
        self.clip_client = clip_client or CLIPClient()
        self.scibert_client = scibert_client or SciBERTClient()
        self.milvus_client = milvus_client or MilvusClient()
        self.logger = logging.getLogger(__name__)

        # Create directories for storing extracted content
        self.figures_dir = "./extracted_content/figures"
        self.tables_dir = "./extracted_content/tables"
        self.pdfs_dir = "./downloaded_pdfs"

        for dir_path in [self.figures_dir, self.tables_dir, self.pdfs_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def download_pdf(self, pdf_url: str, paper_id: str) -> Optional[str]:
        """
        Download PDF from URL and save locally.
        
        Args:
            pdf_url: URL to the PDF file
            paper_id: Paper ID for filename
            
        Returns:
            Local file path if successful, None otherwise
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(pdf_url, headers=headers, timeout=30)
            response.raise_for_status()

            # Save PDF
            pdf_path = os.path.join(self.pdfs_dir, f"{paper_id}.pdf")
            with open(pdf_path, 'wb') as f:
                f.write(response.content)

            self.logger.info(f"Downloaded PDF for paper {paper_id}")
            return pdf_path

        except Exception as e:
            self.logger.error(f"Failed to download PDF for paper {paper_id}: {e}")
            return None

    # ─── Fast file-hash helper (mmap avoids copying large files) ──────

    @staticmethod
    def _fast_file_md5(path: str) -> Optional[str]:
        """Compute MD5 of a file using mmap to avoid loading into Python memory."""
        try:
            with open(path, 'rb') as f:
                size = os.fstat(f.fileno()).st_size
                if size == 0:
                    return hashlib.md5(b'').hexdigest()
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    return hashlib.md5(mm).hexdigest()
        except Exception:
            return None

    # ─── Phase 1: PDF → Markdown conversion (can hang on bad PDFs) ─────

    def convert_pdf_to_markdown(
        self, pdf_path: str, paper_id: str,
    ) -> Optional[Tuple[List[dict], str]]:
        """
        Convert a PDF to markdown page chunks using pymupdf4llm.

        This is the *only* step that can hang on pathological PDFs.
        It is separated so callers can batch-convert many papers first,
        then extract figures/tables from the results.

        Returns:
            ``(page_data, paper_image_dir)`` on success, or ``None``
            if all strategies failed / timed-out.
        """
        paper_image_dir = os.path.join(self.figures_dir, paper_id)
        os.makedirs(paper_image_dir, exist_ok=True)

        # Determine actual page count so we never request out-of-range pages
        try:
            doc = fitz.open(pdf_path)
            actual_pages = doc.page_count
            doc.close()
        except Exception as e:
            self.logger.error(f"Cannot open PDF {pdf_path}: {e}")
            return None

        _max_pages = min(self.MAX_PAGES_TO_PROCESS, actual_pages)
        _common_kwargs = dict(
            pages=list(range(_max_pages)),
            page_chunks=True,
            write_images=True,
            image_path=paper_image_dir,
            image_format="png",
            dpi=150,
            image_size_limit=0.01,
            force_text=True,
            show_progress=False,
        )

        page_data = None
        _strategies = [
            {},
            {"table_strategy": "lines"},
            {"table_strategy": "text"},
            {"ignore_graphics": True},
        ]
        _md_timeout = self.MARKDOWN_TIMEOUT

        for attempt, extra in enumerate(_strategies):
            try:
                md_pool = ThreadPoolExecutor(max_workers=1)
                fut = md_pool.submit(
                    _to_markdown, pdf_path, **{**_common_kwargs, **extra}
                )
                try:
                    page_data = fut.result(timeout=_md_timeout)
                except (FuturesTimeoutError, TimeoutError):
                    self.logger.warning(
                        f"pymupdf4llm attempt {attempt + 1} timed out "
                        f"after {_md_timeout}s for {paper_id} – "
                        f"aborting all strategies (PDF is pathological)"
                    )
                    md_pool.shutdown(wait=False)
                    return None        # bail out immediately
                else:
                    md_pool.shutdown(wait=False)
                    self.logger.info(
                        f"pymupdf4llm converted {paper_id} "
                        f"({len(page_data)} page chunks)"
                    )
                    return (page_data, paper_image_dir)
            except (ValueError, TypeError, AttributeError, IndexError, RuntimeError) as e:
                self.logger.warning(
                    f"pymupdf4llm attempt {attempt + 1} failed for {paper_id}: {e}"
                )
                try:
                    md_pool.shutdown(wait=False)
                except Exception:
                    pass

        self.logger.error(
            f"All pymupdf4llm attempts failed for {paper_id}"
        )
        return None

    # ─── Phase 2: Extract figures & tables from pre-converted data ─────

    def extract_figures_and_tables(
        self, pdf_path: str, paper_id: str,
        page_data: Optional[List[dict]] = None,
        paper_image_dir: Optional[str] = None,
        _deadline: Optional[float] = None,
    ) -> Tuple[List[Figure], List[Table]]:
        """
        Extract figures and tables from a PDF.

        If *page_data* and *paper_image_dir* are supplied (from a prior
        ``convert_pdf_to_markdown`` call), the expensive markdown
        conversion is skipped entirely.  Otherwise the method falls
        back to calling ``convert_pdf_to_markdown`` internally for
        backward compatibility.

        Args:
            pdf_path: Path to the local PDF file.
            paper_id: Unique paper identifier.
            page_data: Pre-computed page chunks from
                ``convert_pdf_to_markdown``.  ``None`` means
                "convert now".
            paper_image_dir: Directory where pymupdf4llm saved the
                extracted images.  Required when *page_data* is given.
            _deadline: Absolute ``time.monotonic()`` deadline.
        """
        figures: List[Figure] = []
        tables: List[Table] = []

        if _deadline is None:
            _deadline = time.monotonic() + self.EXTRACTION_TIMEOUT

        def _timed_out() -> bool:
            return time.monotonic() >= _deadline

        try:
            # ── If page_data was not pre-computed, convert now ────────
            if page_data is None:
                result = self.convert_pdf_to_markdown(pdf_path, paper_id)
                if result is None:
                    return figures, tables
                page_data, paper_image_dir = result
            elif paper_image_dir is None:
                paper_image_dir = os.path.join(self.figures_dir, paper_id)
                os.makedirs(paper_image_dir, exist_ok=True)

            if _timed_out():
                self.logger.warning(
                    f"Timeout after markdown conversion for {paper_id} – skipping"
                )
                return figures, tables

            # ── Phase 1: Collect raw figure/table data (no embeddings yet) ──
            _FigureRaw = type('_FigureRaw', (), {})   # lightweight temp containers
            _TableRaw  = type('_TableRaw', (), {})

            raw_figures: List[object] = []
            raw_tables:  List[object] = []

            figure_counter = 1
            table_counter = 1

            # Deduplication tracking
            _seen_hashes: Set[str] = set()
            _seen_phashes: Set[str] = set()
            _skipped_dupes = 0

            # Local references for hot-loop speed (avoid repeated attribute lookups)
            _max_figs = self.MAX_FIGURES_PER_PAPER
            _max_tabs = self.MAX_TABLES_PER_PAPER
            _max_pages = self.MAX_PAGES_TO_PROCESS
            _img_re = _RE_IMAGE_TAG
            _has_fig_re = _RE_HAS_FIGURE_WORD
            _has_tab_re = _RE_HAS_TABLE_WORD
            _fast_md5 = self._fast_file_md5
            _phash_fn = self._compute_image_phash
            _find_cap_img = self._find_caption_near_image
            _extract_md_tables = self._extract_markdown_tables
            _find_cap_tbl = self._find_caption_near_table
            _parse_md_tbl = self._parse_markdown_table
            _figures_dir = self.figures_dir
            _path_exists = os.path.exists
            _path_join = os.path.join

            for chunk in page_data:
                if len(raw_figures) >= _max_figs and len(raw_tables) >= _max_tabs:
                    break
                if _timed_out():
                    self.logger.warning(
                        f"Timeout during Phase 1 for {paper_id} – "
                        f"collected {len(raw_figures)} figures, {len(raw_tables)} tables so far"
                    )
                    break

                metadata = chunk["metadata"]
                page_num = metadata.get("page") or metadata.get("page_number", 0)
                md_text = chunk.get("text", "")
                if not md_text:
                    continue

                # ── 1. FIGURES ───────────────────────────────────────────
                if len(raw_figures) < _max_figs:
                    img_matches = list(_img_re.finditer(md_text))
                    if img_matches:
                        self.logger.debug(
                            f"Page {page_num}: found {len(img_matches)} image tag(s) in markdown"
                        )
                    for match in img_matches:
                        img_rel_path = match.group(2).strip()

                        # Resolve to absolute path to avoid CWD-dependent failures
                        if not os.path.isabs(img_rel_path):
                            img_rel_path = os.path.abspath(img_rel_path)

                        if not _path_exists(img_rel_path):
                            self.logger.debug(
                                f"Page {page_num}: image path does not exist: {img_rel_path}"
                            )
                            continue

                        # ── Fast file-size pre-check (skip tiny files < 512 B) ──
                        try:
                            fsize = os.path.getsize(img_rel_path)
                            if fsize < 512:
                                self.logger.debug(
                                    f"Page {page_num}: skipping tiny image ({fsize} bytes): {img_rel_path}"
                                )
                                continue
                        except OSError:
                            continue

                        # ── Exact-hash dedup (mmap-based, no Python copy) ──
                        md5 = _fast_md5(img_rel_path)
                        if md5 and md5 in _seen_hashes:
                            _skipped_dupes += 1
                            try:
                                os.remove(img_rel_path)
                            except OSError:
                                pass
                            continue
                        if md5:
                            _seen_hashes.add(md5)

                        # ── Open PIL only after hash check passes ──
                        try:
                            pil_image = Image.open(img_rel_path)
                            w, h = pil_image.size
                            if w < 100 or h < 100:
                                self.logger.debug(
                                    f"Page {page_num}: skipping small image ({w}x{h}): {img_rel_path}"
                                )
                                pil_image.close()
                                continue
                            pil_image = pil_image.convert("RGB")
                        except Exception as img_err:
                            self.logger.debug(
                                f"Page {page_num}: failed to open image: {img_rel_path}: {img_err}"
                            )
                            continue

                        # ── Perceptual-hash dedup ──
                        phash = _phash_fn(pil_image)
                        if phash and phash in _seen_phashes:
                            _skipped_dupes += 1
                            pil_image.close()
                            try:
                                os.remove(img_rel_path)
                            except OSError:
                                pass
                            continue
                        if phash:
                            _seen_phashes.add(phash)

                        description = _find_cap_img(
                            md_text, match.start(),
                            prefix_pattern=r'(?:fig(?:ure|\.)?)\s*\.?\s*(?:S?\d+[a-z]?(?:\([a-z]\))?)',
                            window=2000,
                        )

                        if not description or not _has_fig_re.search(description):
                            # Try a broader search across the entire page text
                            page_level_caption = None
                            if _has_fig_re.search(md_text):
                                page_level_caption = _find_cap_img(
                                    md_text, match.start(),
                                    prefix_pattern=r'(?:fig(?:ure|\.)?)\s*\.?\s*(?:S?\d+[a-z]?(?:\([a-z]\))?)',
                                    window=len(md_text),
                                )
                            if page_level_caption and _has_fig_re.search(page_level_caption):
                                description = page_level_caption
                                self.logger.debug(
                                    f"Page {page_num}: found caption via full-page search: "
                                    f"{description[:80]}..."
                                )
                            else:
                                # Keep the image with a generic description
                                description = f"Figure from page {page_num} of paper {paper_id}"
                                self.logger.info(
                                    f"Page {page_num}: no formal caption found for image "
                                    f"{img_rel_path}, keeping with generic description"
                                )

                        figure_id = f"{paper_id}#figure_{figure_counter}"
                        canonical_path = _path_join(_figures_dir, f"{figure_id}.png")
                        try:
                            if img_rel_path != canonical_path:
                                os.rename(img_rel_path, canonical_path)
                        except OSError:
                            canonical_path = img_rel_path

                        rf = _FigureRaw()
                        rf.figure_id = figure_id
                        rf.paper_id = paper_id
                        rf.figure_number = figure_counter
                        rf.description = description
                        rf.page_number = page_num
                        rf.image_path = canonical_path
                        rf.pil_image = pil_image
                        rf.phash = phash          # cache for dedup phase
                        raw_figures.append(rf)
                        figure_counter += 1
                        if len(raw_figures) >= _max_figs:
                            break

                # ── 2. TABLES ────────────────────────────────────────────
                if len(raw_tables) >= _max_tabs:
                    continue

                table_blocks = _extract_md_tables(md_text)
                if table_blocks:
                    self.logger.debug(
                        f"Page {page_num}: found {len(table_blocks)} markdown table block(s)"
                    )
                for table_md in table_blocks:
                    table_caption = _find_cap_tbl(
                        md_text, table_md,
                        prefix_pattern=r'(?:tab(?:le|\.)?)\s*\.?\s*(?:S?\d+[a-z]?)',
                        window=1200,
                    )

                    headers, rows = _parse_md_tbl(table_md)
                    if not headers and not rows:
                        self.logger.debug(
                            f"Page {page_num}: table block failed parsing (no headers/rows)"
                        )
                        continue

                    table_id = f"{paper_id}#table_{table_counter}"
                    description = table_caption or None

                    if not description or not _has_tab_re.search(description):
                        # Try broader search across full page text
                        if _has_tab_re.search(md_text):
                            page_level_caption = _find_cap_tbl(
                                md_text, table_md,
                                prefix_pattern=r'(?:tab(?:le|\.)?)\s*\.?\s*(?:S?\d+[a-z]?)',
                                window=len(md_text),
                            )
                            if page_level_caption and _has_tab_re.search(page_level_caption):
                                description = page_level_caption
                                self.logger.debug(
                                    f"Page {page_num}: found table caption via full-page search"
                                )
                        if not description or not _has_tab_re.search(description):
                            # Keep table with a generic description
                            description = (
                                f"Table from page {page_num} of paper {paper_id} "
                                f"({len(headers or [])} columns, {len(rows or [])} rows)"
                            )
                            self.logger.info(
                                f"Page {page_num}: no formal caption for table, "
                                f"keeping with generic description"
                            )

                    rt = _TableRaw()
                    rt.table_id = table_id
                    rt.paper_id = paper_id
                    rt.table_number = table_counter
                    rt.description = description
                    rt.page_number = page_num
                    rt.headers = headers
                    rt.rows = rows
                    rt.table_md = table_md
                    raw_tables.append(rt)
                    table_counter += 1
                    if len(raw_tables) >= _max_tabs:
                        break

            self.logger.info(
                f"Phase 1 collection for {paper_id}: "
                f"{len(raw_figures)} figures, {len(raw_tables)} tables "
                f"from {len(page_data)} page chunks "
                f"(dupes skipped: {_skipped_dupes})"
            )

            # ── Open PDF once — reused for native fallback + table rendering ──
            doc = fitz.open(pdf_path)

            # ── Phase 1a-fallback: PyMuPDF native image extraction ──────
            if len(raw_figures) < _max_figs and not _timed_out():
                try:
                    full_text_by_page: Dict[int, str] = {}
                    _fallback_page_limit = min(doc.page_count, _max_pages)

                    for page_idx in range(_fallback_page_limit):
                        if len(raw_figures) >= _max_figs:
                            break

                        page = doc[page_idx]
                        page_num = page_idx + 1

                        image_list = page.get_images(full=True)
                        if not image_list:
                            continue  # skip pages with no images at all

                        # Cache page text lazily (only when page has images)
                        if page_num not in full_text_by_page:
                            full_text_by_page[page_num] = page.get_text("text")

                        page_text = full_text_by_page[page_num]

                        # Pre-check: if the page text has no figure reference,
                        # skip the entire page (saves per-image work)
                        if not _has_fig_re.search(page_text):
                            continue

                        # Find caption once per page, not once per image
                        page_caption = self._find_caption_in_page_text(
                            page_text,
                            prefix_pattern=r'(?:fig(?:ure|\.)?)\s*\.?\s*(?:S?\d+[a-z]?(?:\([a-z]\))?)',
                        )
                        if not page_caption or not _has_fig_re.search(page_caption):
                            continue

                        for img_info in image_list:
                            if len(raw_figures) >= _max_figs:
                                break

                            xref = img_info[0]
                            try:
                                base_image = doc.extract_image(xref)
                                if not base_image or not base_image.get("image"):
                                    continue

                                img_bytes = base_image["image"]

                                # Exact-hash dedup
                                md5 = hashlib.md5(img_bytes).hexdigest()
                                if md5 in _seen_hashes:
                                    _skipped_dupes += 1
                                    continue
                                _seen_hashes.add(md5)

                                pil_image = Image.open(io.BytesIO(img_bytes))
                                w, h = pil_image.size
                                if w < 100 or h < 100:
                                    pil_image.close()
                                    continue
                                pil_image = pil_image.convert("RGB")

                                # Perceptual-hash dedup
                                phash = _phash_fn(pil_image)
                                if phash and phash in _seen_phashes:
                                    _skipped_dupes += 1
                                    pil_image.close()
                                    continue
                                if phash:
                                    _seen_phashes.add(phash)

                                figure_id = f"{paper_id}#figure_{figure_counter}"
                                canonical_path = _path_join(
                                    _figures_dir, f"{figure_id}.png"
                                )
                                pil_image.save(canonical_path, "PNG")

                                rf = _FigureRaw()
                                rf.figure_id = figure_id
                                rf.paper_id = paper_id
                                rf.figure_number = figure_counter
                                rf.description = page_caption
                                rf.page_number = page_num
                                rf.image_path = canonical_path
                                rf.pil_image = pil_image
                                rf.phash = phash          # cache for dedup phase
                                raw_figures.append(rf)
                                figure_counter += 1

                            except Exception as img_err:
                                self.logger.debug(
                                    f"Could not extract image xref={xref} "
                                    f"from page {page_num}: {img_err}"
                                )
                                continue

                    if figure_counter > 1:
                        self.logger.info(
                            f"PyMuPDF native fallback found {len(raw_figures)} "
                            f"total figures for {paper_id}"
                        )
                except Exception as e:
                    self.logger.warning(f"PyMuPDF native figure fallback failed: {e}")

            # ── Phase 1b: Remove sub-images contained in larger figures ─
            if not _timed_out():
                raw_figures, containment_removed = self._remove_contained_sub_images(raw_figures)
                _skipped_dupes += containment_removed
            else:
                containment_removed = 0
                self.logger.warning(f"Timeout before dedup for {paper_id} – skipping sub-image removal")

            if _skipped_dupes > 0:
                self.logger.info(
                    f"Dedup: skipped {_skipped_dupes} duplicate/sub-images for {paper_id}"
                )

            # ── Phase 1c: Cap figures & tables to configured limits ──────
            if len(raw_figures) > _max_figs:
                # Keep the largest images (by pixel area) — they are the
                # most likely to be full figures rather than icons/logos.
                raw_figures.sort(
                    key=lambda rf: rf.pil_image.width * rf.pil_image.height,
                    reverse=True,
                )
                excess = raw_figures[_max_figs:]
                raw_figures = raw_figures[:_max_figs]
                for rf in excess:
                    try:
                        os.remove(rf.image_path)
                    except OSError:
                        pass
                # Re-sort by page order and re-number
                raw_figures.sort(key=lambda rf: (rf.page_number, rf.figure_number))
                for i, rf in enumerate(raw_figures, 1):
                    new_id = f"{rf.paper_id}#figure_{i}"
                    if rf.figure_id != new_id:
                        new_path = _path_join(
                            os.path.dirname(rf.image_path), f"{new_id}.png"
                        )
                        try:
                            if _path_exists(rf.image_path):
                                os.rename(rf.image_path, new_path)
                            rf.image_path = new_path
                        except OSError:
                            pass
                        rf.figure_id = new_id
                    rf.figure_number = i
                self.logger.info(
                    f"Capped figures from {len(raw_figures) + len(excess)} "
                    f"to {_max_figs} for {paper_id}"
                )

            if len(raw_tables) > _max_tabs:
                raw_tables = raw_tables[:_max_tabs]
                for i, rt in enumerate(raw_tables, 1):
                    rt.table_id = f"{rt.paper_id}#table_{i}"
                    rt.table_number = i
                self.logger.info(
                    f"Capped tables to {_max_tabs} for {paper_id}"
                )

            # ── Phase 2: Render table images (reuse already-open doc) ────
            table_images: List[Tuple[Optional[str], Optional[Image.Image]]] = []
            if raw_tables and not _timed_out():
                try:
                    for rt in raw_tables:
                        img_path, pil_img = self._render_table_image_from_doc(
                            doc, rt.page_number - 1, rt.table_md, rt.table_id
                        )
                        table_images.append((img_path, pil_img))
                except Exception as e:
                    self.logger.error(f"Error rendering table images: {e}")
                    while len(table_images) < len(raw_tables):
                        table_images.append((None, None))

            # Close the single PDF document
            doc.close()

            if _timed_out():
                self.logger.warning(
                    f"Timeout before embeddings for {paper_id} – "
                    f"returning {len(raw_figures)} figures, {len(raw_tables)} tables without embeddings"
                )
                # Build entities without embeddings so we don't lose the extraction work
                for rf in raw_figures:
                    figures.append(Figure(
                        id=rf.figure_id, paper_id=rf.paper_id,
                        figure_number=rf.figure_number, description=rf.description,
                        page_number=rf.page_number, image_path=rf.image_path,
                        image_embedding=None, description_embedding=None,
                    ))
                for i, rt in enumerate(raw_tables):
                    img_path = table_images[i][0] if i < len(table_images) else None
                    tables.append(Table(
                        id=rt.table_id, paper_id=rt.paper_id,
                        table_number=rt.table_number, description=rt.description,
                        page_number=rt.page_number, headers=rt.headers, rows=rt.rows,
                        description_embedding=None, image_embedding=None,
                    ))
                for rf in raw_figures:
                    rf.pil_image = None
                return figures, tables

            # ── Phase 3: Batch embeddings (CLIP + SciBERT concurrent) ────
            fig_pil_images   = [rf.pil_image for rf in raw_figures]
            fig_descriptions = [rf.description for rf in raw_figures]

            tbl_descriptions = [rt.description for rt in raw_tables]
            tbl_pil_images   = [ti[1] for ti in table_images]

            all_clip_images: List[Optional[Image.Image]] = fig_pil_images + tbl_pil_images
            all_scibert_texts = fig_descriptions + tbl_descriptions

            # Run CLIP (image) and SciBERT (text) embedding batches
            # concurrently — both release the GIL during CUDA kernels.
            all_clip_embs: list = []
            all_scibert_embs: list = []

            def _run_clip():
                nonlocal all_clip_embs
                if all_clip_images:
                    all_clip_embs = self.clip_client.generate_image_embeddings_batch(
                        list(all_clip_images),
                    )

            def _run_scibert():
                nonlocal all_scibert_embs
                if all_scibert_texts:
                    all_scibert_embs = self.scibert_client.generate_text_embeddings_batch(
                        all_scibert_texts
                    )

            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_clip = pool.submit(_run_clip)
                fut_scibert = pool.submit(_run_scibert)
                fut_clip.result()
                fut_scibert.result()

            fig_clip_embs = all_clip_embs[:len(raw_figures)]
            tbl_clip_embs = all_clip_embs[len(raw_figures):]
            fig_scibert_embs = all_scibert_embs[:len(raw_figures)]
            tbl_scibert_embs = all_scibert_embs[len(raw_figures):]

            # ── Phase 4: Build final entities ────────────────────────────
            for i, rf in enumerate(raw_figures):
                figures.append(Figure(
                    id=rf.figure_id,
                    paper_id=rf.paper_id,
                    figure_number=rf.figure_number,
                    description=rf.description,
                    page_number=rf.page_number,
                    image_path=rf.image_path,
                    image_embedding=fig_clip_embs[i] if i < len(fig_clip_embs) else None,
                    description_embedding=fig_scibert_embs[i] if i < len(fig_scibert_embs) else None,
                ))

            for i, rt in enumerate(raw_tables):
                img_path = table_images[i][0] if i < len(table_images) else None
                tables.append(Table(
                    id=rt.table_id,
                    paper_id=rt.paper_id,
                    table_number=rt.table_number,
                    description=rt.description,
                    page_number=rt.page_number,
                    headers=rt.headers,
                    rows=rt.rows,
                    description_embedding=tbl_scibert_embs[i] if i < len(tbl_scibert_embs) else None,
                    image_embedding=tbl_clip_embs[i] if i < len(tbl_clip_embs) else None,
                ))

            # Free references to PIL images
            for rf in raw_figures:
                rf.pil_image = None

            # Clean up leftover pymupdf4llm temp images in background thread
            # Only remove files that were NOT moved to the canonical figures_dir
            extracted_paths = {fig.image_path for fig in figures}
            def _cleanup_temp_dir(d: str, keep_paths: Set[str]):
                try:
                    for leftover in os.listdir(d):
                        p = os.path.join(d, leftover)
                        if os.path.isfile(p) and p not in keep_paths:
                            os.remove(p)
                    # Only remove dir if empty
                    if not os.listdir(d):
                        os.rmdir(d)
                except OSError:
                    pass

            ThreadPoolExecutor(max_workers=1).submit(
                _cleanup_temp_dir, paper_image_dir, extracted_paths
            )

            self.logger.info(
                f"Extracted {len(figures)} figures and {len(tables)} tables "
                f"from {paper_id} using pymupdf4llm (batch embeddings)"
            )

        except Exception as e:
            self.logger.error(f"Critical error processing PDF {pdf_path}: {e}", exc_info=True)

        return figures, tables

    # ─── Deduplication helpers ──────────────────────────────────────────

    @staticmethod
    def _compute_image_phash(pil_image: Image.Image, hash_size: int = 8) -> Optional[str]:
        """
        Compute a perceptual hash (pHash) for an image.

        Returns a hex string, or None on error.
        """
        try:
            return format(
                PDFProcessingHandler._compute_image_phash_int(pil_image, hash_size),
                f'0{(hash_size * 4) ** 2 // 4}x',
            )
        except Exception:
            return None

    @staticmethod
    def _compute_image_phash_int(pil_image: Image.Image, hash_size: int = 8) -> int:
        """
        Compute a perceptual hash as a raw integer.

        Uses BILINEAR resize and ``int.from_bytes`` on a pre-built
        bytearray — avoids the slow Python ``for b in raw`` loop
        over 1024 bytes entirely.

        Raises on error (caller should catch).
        """
        dim = hash_size * 4  # 32 for default hash_size=8
        raw = pil_image.resize((dim, dim), Image.BILINEAR).convert("L").tobytes()
        n = len(raw)
        avg = sum(raw) / n
        # Build a bytes object where each byte is 0 or 1, then convert
        # to int in one shot — ~50× faster than a Python for-loop.
        bits = bytes(1 if b > avg else 0 for b in raw)
        return int.from_bytes(bits, 'big')  # wrong magnitude but consistent

    @staticmethod
    def _phash_distance(h1: str, h2: str) -> int:
        """Hamming distance between two hex-encoded hashes."""
        if len(h1) != len(h2):
            return 999
        b1 = int(h1, 16)
        b2 = int(h2, 16)
        xor = b1 ^ b2
        return bin(xor).count('1')

    def _remove_contained_sub_images(
        self, raw_figures: List[object], area_ratio_threshold: float = 0.35
    ) -> Tuple[List[object], int]:
        """
        Remove sub-images that are visually contained in a larger figure
        from the same page, and near-identical duplicates.

        Optimisations over the naïve approach:
        - Integer pHash cached on each ``_FigureRaw`` during Phase 1 via
          ``_compute_image_phash_int`` — no hex string round-trip.
        - Flat ``bytearray`` (not ``set``) for removed-tracking — O(1)
          index access without hashing overhead.
        - Inner loop uses pre-fetched local variables and direct index
          access into flat arrays (no ``getattr``, no dict lookup).
        - All filesystem I/O (``os.remove``, ``os.rename``) is deferred
          to the very end so it never blocks the comparison loop.
        - Single-figure pages short-circuited before any allocation.

        Returns (filtered_figures, num_removed).
        """
        n_total = len(raw_figures)
        if n_total <= 1:
            return raw_figures, 0

        # ── Group by page ────────────────────────────────────────────
        by_page: Dict[int, List[int]] = {}   # page -> indices into raw_figures
        for idx, rf in enumerate(raw_figures):
            by_page.setdefault(rf.page_number, []).append(idx)

        # Global boolean mask — True means "remove this figure"
        removed_mask = bytearray(n_total)  # all zeros = keep

        # Pre-compute areas + integer pHashes for ALL figures once
        areas = [0] * n_total
        phash_ints = [None] * n_total       # type: List[Optional[int]]
        _phash_int_fn = self._compute_image_phash_int

        for idx, rf in enumerate(raw_figures):
            w, h = rf.pil_image.size
            areas[idx] = w * h
            # Try integer phash cached on the object first
            phi = getattr(rf, '_phash_int', None)
            if phi is not None:
                phash_ints[idx] = phi
                continue
            # Fall back to hex-string cache from Phase 1
            ph = getattr(rf, 'phash', None)
            if ph is not None:
                try:
                    phi = int(ph, 16)
                except (ValueError, TypeError):
                    phi = None
            # Last resort: compute from image
            if phi is None:
                try:
                    phi = _phash_int_fn(rf.pil_image)
                except Exception:
                    phi = None
            phash_ints[idx] = phi
            rf._phash_int = phi  # cache for future use

        # Bind popcount function once
        _bit_count = int.bit_count if hasattr(int, 'bit_count') else None

        # ── Per-page pairwise comparison ─────────────────────────────
        for page_indices in by_page.values():
            k = len(page_indices)
            if k <= 1:
                continue

            # Sort indices by area descending (largest first)
            page_indices.sort(key=lambda i: areas[i], reverse=True)

            # Pull values into contiguous local arrays for tight loop
            p_areas = [areas[i] for i in page_indices]
            p_hashes = [phash_ints[i] for i in page_indices]
            p_removed = bytearray(k)  # local removed flags

            # Tight O(k²) loop — all lookups are array[int] (no dict/set)
            for a in range(k):
                if p_removed[a]:
                    continue
                phi = p_hashes[a]
                if phi is None:
                    continue
                ai = p_areas[a]
                ai_inv_thresh = ai * area_ratio_threshold  # pre-multiply

                for b in range(a + 1, k):
                    if p_removed[b]:
                        continue
                    phj = p_hashes[b]
                    if phj is None:
                        continue

                    xor = phi ^ phj
                    dist = xor.bit_count() if _bit_count else bin(xor).count('1')

                    # Near-identical → duplicate
                    if dist <= 5:
                        p_removed[b] = 1
                        continue

                    # Sub-image containment (b is smaller since sorted desc)
                    if dist <= 20 and p_areas[b] < ai_inv_thresh:
                        p_removed[b] = 1

            # Write local results back to global mask
            for loc_idx in range(k):
                if p_removed[loc_idx]:
                    removed_mask[page_indices[loc_idx]] = 1

        # ── Build filtered list and count removals ───────────────────
        removed = 0
        files_to_delete: List[str] = []
        filtered: List[object] = []

        for idx in range(n_total):
            if removed_mask[idx]:
                removed += 1
                files_to_delete.append(raw_figures[idx].image_path)
            else:
                filtered.append(raw_figures[idx])

        # Re-number surviving figures sequentially
        rename_ops: List[Tuple[str, str]] = []  # (old_path, new_path)
        for i, rf in enumerate(filtered, 1):
            new_id = f"{rf.paper_id}#figure_{i}"
            if rf.figure_id != new_id:
                new_path = os.path.join(
                    os.path.dirname(rf.image_path), f"{new_id}.png"
                )
                rename_ops.append((rf.image_path, new_path))
                rf.image_path = new_path
                rf.figure_id = new_id
            rf.figure_number = i

        # ── Deferred filesystem I/O (never inside the hot loop) ──────
        for path in files_to_delete:
            try:
                os.remove(path)
            except OSError:
                pass
        for old_path, new_path in rename_ops:
            try:
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
            except OSError:
                pass

        return filtered, removed

    # ─── Helper methods for the pymupdf4llm-based extraction ────────────

    @staticmethod
    def _find_caption_near_image(md_text: str, img_pos: int,
                                 prefix_pattern: str = '', window: int = 800) -> Optional[str]:
        """
        Search for a figure caption near the image reference.

        Uses pre-compiled module-level regexes (_RE_FIG_CAP_STD,
        _RE_FIG_CAP_BOLD) — zero per-call compilation cost.
        """
        start = max(0, img_pos - window)
        end = min(len(md_text), img_pos + window)
        neighbourhood = md_text[start:end]

        # Strategy 1: Standard caption
        m = _RE_FIG_CAP_STD.search(neighbourhood)
        if m:
            caption = m.group(1).strip()
            if len(caption) > 1000:
                caption = caption[:1000].rsplit('.', 1)[0] + '.'
            return caption

        # Strategy 2: Bold/italic caption
        m2 = _RE_FIG_CAP_BOLD.search(neighbourhood)
        if m2:
            caption = m2.group(1).strip().strip('*').strip()
            if len(caption) > 1000:
                caption = caption[:1000].rsplit('.', 1)[0] + '.'
            return caption

        return None

    @staticmethod
    def _find_caption_in_page_text(page_text: str,
                                   prefix_pattern: str = '') -> Optional[str]:
        """
        Search raw page text for a figure caption.

        Uses pre-compiled _RE_PAGE_CAP_FIG — zero per-call cost.
        """
        if not page_text:
            return None

        m = _RE_PAGE_CAP_FIG.search(page_text)
        if m:
            caption = m.group(1).strip()
            if len(caption) > 1000:
                caption = caption[:1000].rsplit('.', 1)[0] + '.'
            return caption
        return None

    @staticmethod
    def _is_valid_table(table_lines: List[str]) -> bool:
        """
        Validate that a candidate table block is a genuine GFM markdown table.

        Uses pre-compiled _RE_SEP_CELL for speed.
        """
        if len(table_lines) < 3:
            return False

        # ── Locate separator row (must be line index 1) ──────────────
        sep_cells = [c.strip() for c in table_lines[1].strip('|').split('|')]

        if not all(_RE_SEP_CELL.match(c) for c in sep_cells):
            return False

        num_cols = len(sep_cells)
        if num_cols < 2:
            return False

        # ── Header column count must match separator ─────────────────
        header_cols = table_lines[0].count('|') - 1  # fast column count
        if header_cols < 1:
            header_cols = len([c.strip() for c in table_lines[0].strip('|').split('|')])
        if abs(header_cols - num_cols) > 1:
            return False

        # ── Must have at least 1 data row ────────────────────────────
        data_lines = table_lines[2:]
        if not data_lines:
            return False

        # ── Column-count consistency + data stats in single pass ─────
        mismatches = 0
        non_empty_cells = 0
        total_cell_len = 0
        total_cells = 0
        for dl in data_lines:
            cells = [c.strip() for c in dl.strip('|').split('|')]
            if abs(len(cells) - num_cols) > 1:
                mismatches += 1
            for c in cells:
                total_cells += 1
                clen = len(c)
                total_cell_len += clen
                if clen > 0:
                    non_empty_cells += 1

        if mismatches > len(data_lines) * 0.5:
            return False
        if non_empty_cells == 0:
            return False
        if total_cells > 0 and (total_cell_len / total_cells) > 120:
            return False

        return True

    @staticmethod
    def _extract_markdown_tables(md_text: str) -> List[str]:
        """
        Extract all GFM markdown table blocks from text.

        A valid markdown table must have:
          - A header row
          - A separator row with ``---`` cells (line 2)
          - At least one data row
          - At least 2 columns
        """
        tables = []
        lines = md_text.split('\n')
        current_table_lines: List[str] = []
        in_table = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('|') and stripped.endswith('|'):
                current_table_lines.append(stripped)
                in_table = True
            else:
                if in_table and current_table_lines:
                    if PDFProcessingHandler._is_valid_table(current_table_lines):
                        tables.append('\n'.join(current_table_lines))
                    current_table_lines = []
                in_table = False

        # Handle table at end of text
        if current_table_lines:
            if PDFProcessingHandler._is_valid_table(current_table_lines):
                tables.append('\n'.join(current_table_lines))

        return tables

    @staticmethod
    def _find_caption_near_table(md_text: str, table_md: str,
                                 prefix_pattern: str = '', window: int = 600) -> Optional[str]:
        """
        Find a table caption near the table block.

        Uses pre-compiled _RE_TBL_CAP_STD / _RE_TBL_CAP_BOLD —
        zero per-call compilation cost.
        """
        pos = md_text.find(table_md)
        if pos == -1:
            return None
        start = max(0, pos - window)
        end = min(len(md_text), pos + len(table_md) + window)
        neighbourhood = md_text[start:end]

        # Strategy 1: standard caption
        m = _RE_TBL_CAP_STD.search(neighbourhood)
        if m:
            caption = m.group(1).strip()
            if len(caption) > 1000:
                caption = caption[:1000].rsplit('.', 1)[0] + '.'
            return caption

        # Strategy 2: bold/italic caption
        m2 = _RE_TBL_CAP_BOLD.search(neighbourhood)
        if m2:
            caption = m2.group(1).strip().strip('*').strip()
            if len(caption) > 1000:
                caption = caption[:1000].rsplit('.', 1)[0] + '.'
            return caption

        return None

    @staticmethod
    def _parse_markdown_table(table_md: str) -> Tuple[Optional[List[str]], Optional[List[List[str]]]]:
        """
        Parse a GFM markdown table into headers and rows.

        Expected format::

            | Model | Acc | F1 |
            |-------|-----|-----|
            | BERT  | 0.9 | 0.8 |

        Returns (None, None) for invalid input.
        """
        lines = [l.strip() for l in table_md.strip().split('\n') if l.strip()]
        if len(lines) < 3:               # header + separator + ≥1 data row
            return None, None

        def _split_row(line: str) -> List[str]:
            return [c.strip() for c in line.strip('|').split('|')]

        # Line 0 = header, line 1 = separator, lines 2+ = data
        headers = _split_row(lines[0])

        rows = []
        for line in lines[2:]:            # skip header (0) and separator (1)
            if _RE_EXTRA_SEP_ROW.match(line):
                continue                  # extra separator row
            rows.append(_split_row(line))

        return headers, rows if rows else None

    def _render_table_image_from_doc(self, doc: fitz.Document, page_index: int,
                                     table_md: str, table_id: str
                                     ) -> Tuple[Optional[str], Optional[Image.Image]]:
        """
        Render a table region from an already-open PDF document as a PNG.

        Strategy order:
        1. Use PyMuPDF ``page.find_tables()`` — gives pixel-perfect bboxes.
           Match our markdown table to the best native table by header overlap.
        2. Fallback: text-search for header AND last-row cells to anchor
           top-left and bottom-right corners precisely.
        3. Last resort: render entire page (rare).

        Returns (image_path, pil_image).
        """
        try:
            if page_index >= doc.page_count or page_index < 0:
                return None, None

            page = doc[page_index]

            lines = [l.strip() for l in table_md.strip().split('\n') if l.strip()]
            if not lines:
                return None, None

            header_cells = [c.strip() for c in lines[0].strip('|').split('|') if c.strip()]
            # Last data row (skip separator lines)
            data_lines = [l for l in lines[2:] if not re.match(r'^[\s|:-]+$', l)]
            last_row_cells = []
            if data_lines:
                last_row_cells = [c.strip() for c in data_lines[-1].strip('|').split('|') if c.strip()]

            table_rect = None

            # ── Strategy 1: PyMuPDF native table detection ──────────────
            try:
                tabs = page.find_tables()
                if tabs and tabs.tables:
                    header_set = {c.lower() for c in header_cells if c}
                    best_table = None
                    best_overlap = 0
                    for tab in tabs.tables:
                        # tab.header.names contains the detected header cells
                        try:
                            native_headers = set()
                            if hasattr(tab, 'header') and hasattr(tab.header, 'names'):
                                native_headers = {
                                    str(n).strip().lower()
                                    for n in tab.header.names if n
                                }
                            # Also try first row of extracted content
                            if not native_headers and hasattr(tab, 'extract'):
                                rows_data = tab.extract()
                                if rows_data and rows_data[0]:
                                    native_headers = {
                                        str(c).strip().lower()
                                        for c in rows_data[0] if c
                                    }
                            overlap = len(header_set & native_headers)
                            if overlap > best_overlap:
                                best_overlap = overlap
                                best_table = tab
                        except Exception:
                            continue

                    if best_table and best_overlap >= max(1, len(header_set) // 2):
                        table_rect = fitz.Rect(best_table.bbox)
                        self.logger.debug(
                            f"Table {table_id}: native bbox found "
                            f"(overlap={best_overlap}/{len(header_set)})"
                        )
            except Exception as e:
                self.logger.debug(f"find_tables() failed for {table_id}: {e}")

            # ── Strategy 2: Text-search anchoring (header + last row) ───
            if table_rect is None:
                top_rect = None
                bottom_rect = None

                # Search for ALL header cells to get a precise top edge
                for cell_text in header_cells:
                    if not cell_text or len(cell_text) < 2:
                        continue
                    hits = page.search_for(cell_text)
                    if hits:
                        # Pick the hit closest to existing top_rect, or first
                        hit = hits[0]
                        if top_rect is None:
                            top_rect = hit
                        else:
                            top_rect = top_rect | hit

                # Search for last data-row cells to anchor bottom edge
                for cell_text in last_row_cells:
                    if not cell_text or len(cell_text) < 2:
                        continue
                    hits = page.search_for(cell_text)
                    if hits:
                        # Pick the lowest hit on the page
                        hit = max(hits, key=lambda r: r.y1)
                        if bottom_rect is None:
                            bottom_rect = hit
                        else:
                            bottom_rect = bottom_rect | hit

                if top_rect is not None:
                    if bottom_rect is not None:
                        # Merge top and bottom anchors
                        table_rect = top_rect | bottom_rect
                    else:
                        # No bottom anchor — estimate from row count
                        num_data_rows = len(data_lines)
                        table_rect = fitz.Rect(top_rect)
                        table_rect.y1 = min(
                            page.rect.height,
                            table_rect.y1 + num_data_rows * 18,
                        )

            # ── Apply padding and clamp ─────────────────────────────────
            if table_rect is not None and not table_rect.is_empty:
                padding = 15
                table_rect.x0 = max(0, table_rect.x0 - padding)
                table_rect.y0 = max(0, table_rect.y0 - padding)
                table_rect.x1 = min(page.rect.width, table_rect.x1 + padding)
                table_rect.y1 = min(page.rect.height, table_rect.y1 + padding)

                # Sanity: if the rect is >85% of page area, it's likely wrong
                page_area = page.rect.width * page.rect.height
                table_area = table_rect.width * table_rect.height
                if page_area > 0 and (table_area / page_area) > 0.85:
                    self.logger.debug(
                        f"Table {table_id}: bbox covers {table_area/page_area:.0%} "
                        f"of page — falling back to full page"
                    )
                    table_rect = page.rect
            else:
                # Strategy 3: full page fallback
                table_rect = page.rect

            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat, clip=table_rect)

            img_data = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_data)).convert("RGB")

            image_path = os.path.join(self.tables_dir, f"{table_id}_image.png")
            pil_image.save(image_path)

            pix = None
            return image_path, pil_image

        except Exception as e:
            self.logger.error(f"Error rendering table image for {table_id}: {e}")
            return None, None

    def _render_table_image_from_page(self, pdf_path: str, page_index: int,
                                      table_md: str, table_id: str
                                      ) -> Tuple[Optional[str], Optional[List[float]]]:
        """
        Legacy single-call variant: opens the PDF, renders, and embeds.

        Delegates to ``_render_table_image_from_doc`` for the actual
        bbox detection, then generates a CLIP embedding.
        """
        try:
            doc = fitz.open(pdf_path)
            img_path, pil_image = self._render_table_image_from_doc(
                doc, page_index, table_md, table_id
            )
            doc.close()

            image_embedding = None
            if pil_image is not None:
                try:
                    image_embedding = self.clip_client.generate_image_embedding(pil_image)
                except Exception as embed_err:
                    self.logger.warning(f"Failed embedding for table {table_id}: {embed_err}")

            return img_path, image_embedding

        except Exception as e:
            self.logger.error(f"Error rendering table image for {table_id}: {e}")
            return None, None

    def _save_figures_to_milvus(self, figures: List[Figure]) -> bool:
        """
        Save figure embeddings to Milvus using dedicated figures collection.
        
        Args:
            figures: List of Figure entities with embeddings
            
        Returns:
            True if successful, False otherwise
        """
        if not figures:
            return True

        try:
            # Fit TF-IDF vectorizer once for all descriptions
            if not self.milvus_client.is_tfidf_fitted:
                all_descriptions = [fig.description for fig in figures if fig.description]
                if all_descriptions:
                    self.milvus_client.fit_tfidf_vectorizer(all_descriptions)
                    self.logger.info(f"Fitted TF-IDF vectorizer with {len(all_descriptions)} figure descriptions")

            # Prepare figures data for the dedicated figures collection
            figures_data = []
            for figure in figures:
                # Only upload figures that have at least one type of embedding
                if figure.description_embedding is not None or figure.image_embedding is not None:
                    # Generate sparse embedding for description if available
                    sparse_description_embedding = {0: 0.001}  # Default minimal sparse vector
                    if figure.description and self.milvus_client.is_tfidf_fitted:
                        sparse_description_embedding = self.milvus_client.generate_sparse_embedding(figure.description)

                    # Ensure embeddings are valid lists, not None
                    description_embedding = figure.description_embedding if figure.description_embedding is not None else [0.0] * 768
                    image_embedding = figure.image_embedding if figure.image_embedding is not None else [0.0] * 768

                    figure_data = {
                        'id': figure.id,
                        'paper_id': figure.paper_id,
                        'description': figure.description or "",
                        'description_embedding': description_embedding,
                        'image_embedding': image_embedding,
                        'sparse_description_embedding': sparse_description_embedding
                    }
                    figures_data.append(figure_data)

            if not figures_data:
                self.logger.warning("No figure embeddings to save to Milvus")
                return True

            # Upload using the dedicated figures collection method
            success = self.milvus_client.upload_figures_embeddings(
                figures_data=figures_data,
                batch_size=100
            )

            if success:
                self.logger.info(
                    f"Successfully saved {len(figures_data)} figures to dedicated Milvus figures collection")
            else:
                self.logger.error("Failed to save figures to Milvus")

            return success

        except Exception as e:
            self.logger.error(f"Error saving figures to Milvus: {e}")
            return False

    def _save_tables_to_milvus(self, tables: List[Table]) -> bool:
        """
        Save table embeddings to Milvus using dedicated tables collection.
        
        Args:
            tables: List of Table entities with embeddings
            
        Returns:
            True if successful, False otherwise
        """
        if not tables:
            return True

        try:
            # Fit TF-IDF vectorizer once for all descriptions and table texts
            if not self.milvus_client.is_tfidf_fitted:
                all_texts = []
                for table in tables:
                    # Combine description and table text for richer embeddings
                    # combined_text = f"{table.description or ''} {table.table_text or ''}".strip()
                    if table.description:
                        all_texts.append(table.description)

                if all_texts:
                    self.milvus_client.fit_tfidf_vectorizer(all_texts)
                    self.logger.info(f"Fitted TF-IDF vectorizer with {len(all_texts)} table texts")

            # Prepare tables data for the dedicated tables collection
            tables_data = []
            for table in tables:
                # Only upload tables that have valid description embeddings
                if table.description_embedding is not None:
                    # Ensure embedding is a valid list, not None
                    description_embedding = table.description_embedding if table.description_embedding is not None else [0.0] * 768

                    table_data = {
                        'id': table.id,
                        'paper_id': table.paper_id,
                        'description': table.description or "",
                        'description_embedding': description_embedding,
                        'image_embedding': table.image_embedding,
                    }
                    tables_data.append(table_data)

            if not tables_data:
                self.logger.warning("No table embeddings to save to Milvus")
                return True

            # Upload using the dedicated tables collection method
            success = self.milvus_client.upload_tables_embeddings(
                tables_data=tables_data,
                batch_size=100
            )

            if success:
                self.logger.info(f"Successfully saved {len(tables_data)} tables to dedicated Milvus tables collection")
            else:
                self.logger.error("Failed to save tables to Milvus")

            return success

        except Exception as e:
            self.logger.error(f"Error saving tables to Milvus: {e}")
            return False

    # ─── Single-paper convenience (backward-compatible) ─────────────

    def process_paper_pdf(self, paper: Paper,
                          timeout: Optional[float] = None) -> Tuple[List[Figure], List[Table]]:
        """
        Complete pipeline to process a *single* paper's PDF.

        For batch processing prefer ``process_papers_batch`` which
        separates the risky markdown-conversion step from the
        extraction + upload step.
        """
        if not paper.pdf_url:
            self.logger.warning(f"No PDF URL provided for paper {paper.id}")
            return [], []

        if timeout is None:
            timeout = self.EXTRACTION_TIMEOUT

        t_start = time.monotonic()
        deadline = t_start + timeout

        # Download PDF
        pdf_path = self.download_pdf(paper.pdf_url, paper.id)
        if not pdf_path:
            return [], []

        elapsed_download = time.monotonic() - t_start
        if elapsed_download >= timeout:
            self.logger.warning(
                f"Paper {paper.id}: download alone took {elapsed_download:.1f}s "
                f"(limit {timeout}s) – skipping extraction"
            )
            try:
                os.remove(pdf_path)
            except OSError:
                pass
            return [], []

        # Phase 1 – convert to markdown (with hard per-strategy timeout)
        md_result = self.convert_pdf_to_markdown(pdf_path, paper.id)

        if md_result is None:
            try:
                os.remove(pdf_path)
            except OSError:
                pass
            return [], []

        page_data, paper_image_dir = md_result

        # Phase 2 – extract figures/tables + embeddings
        figures, tables = self.extract_figures_and_tables(
            pdf_path, paper.id,
            page_data=page_data,
            paper_image_dir=paper_image_dir,
            _deadline=deadline,
        )

        elapsed_extract = time.monotonic() - t_start
        if elapsed_extract >= timeout:
            self.logger.warning(
                f"Paper {paper.id}: extraction took {elapsed_extract:.1f}s "
                f"(limit {timeout}s) – skipping Milvus upload"
            )
            try:
                os.remove(pdf_path)
            except OSError:
                pass
            return figures, tables

        # Phase 3 – upload to Zilliz/Milvus
        self._upload_to_milvus(figures, tables)

        elapsed_total = time.monotonic() - t_start
        self.logger.info(f"Paper {paper.id}: total processing time {elapsed_total:.1f}s")

        try:
            os.remove(pdf_path)
        except OSError:
            pass

        return figures, tables

    # ─── Batch processing (Phase 1 → Phase 2 for all papers) ──────────

    def process_papers_batch(
        self, papers: List[Paper],
        timeout_per_paper: Optional[float] = None,
    ) -> List[Tuple[List[Figure], List[Table]]]:
        """
        Two-phase batch pipeline:

        **Phase 1 – Download + Markdown** (for *all* papers):  
        Download each PDF and run ``convert_pdf_to_markdown``.  
        Papers whose conversion times out or fails are skipped
        gracefully — they never block the rest of the batch.

        **Phase 2 – Extract + Embed + Upload** (for *all* successful papers):  
        For every paper that produced valid ``page_data``, extract
        figures/tables, generate CLIP + SciBERT embeddings, and
        upload to Zilliz/Milvus.

        Args:
            papers: List of ``Paper`` entities with ``pdf_url``.
            timeout_per_paper: Max seconds for extraction per paper.
                Defaults to ``EXTRACTION_TIMEOUT``.

        Returns:
            List of ``(figures, tables)`` tuples, one per input paper
            (empty lists for papers that were skipped).
        """
        if timeout_per_paper is None:
            timeout_per_paper = self.EXTRACTION_TIMEOUT

        n = len(papers)
        results: List[Tuple[List[Figure], List[Table]]] = [([], [])] * n

        # ── Phase 1: Download + convert all papers ───────────────────
        # Stores (pdf_path, page_data, paper_image_dir) per paper index
        converted: Dict[int, Tuple[str, List[dict], str]] = {}

        self.logger.info(f"Phase 1: Converting {n} PDFs to markdown...")
        for idx, paper in enumerate(papers):
            if not paper.pdf_url:
                self.logger.warning(
                    f"[{idx+1}/{n}] {paper.id}: no PDF URL – skipping"
                )
                continue

            pdf_path = self.download_pdf(paper.pdf_url, paper.id)
            if not pdf_path:
                continue

            try:
                md_result = self.convert_pdf_to_markdown(pdf_path, paper.id)
            except Exception as e:
                self.logger.error(
                    f"[{idx+1}/{n}] {paper.id}: convert_pdf_to_markdown crashed: {e}"
                )
                md_result = None

            if md_result is None:
                self.logger.warning(
                    f"[{idx+1}/{n}] {paper.id}: markdown conversion failed – skipping"
                )
                try:
                    os.remove(pdf_path)
                except OSError:
                    pass
                continue

            page_data, paper_image_dir = md_result
            converted[idx] = (pdf_path, page_data, paper_image_dir)
            self.logger.info(
                f"[{idx+1}/{n}] {paper.id}: converted "
                f"({len(page_data)} page chunks)"
            )

        self.logger.info(
            f"Phase 1 complete: {len(converted)}/{n} papers converted successfully"
        )

        # ── Phase 2: Extract figures/tables + embed + upload ─────────
        self.logger.info(
            f"Phase 2: Extracting figures/tables from {len(converted)} papers..."
        )
        total_figures = 0
        total_tables = 0

        for idx, (pdf_path, page_data, paper_image_dir) in converted.items():
            paper = papers[idx]
            deadline = time.monotonic() + timeout_per_paper

            try:
                figures, tables = self.extract_figures_and_tables(
                    pdf_path, paper.id,
                    page_data=page_data,
                    paper_image_dir=paper_image_dir,
                    _deadline=deadline,
                )

                # Upload to Zilliz/Milvus
                self._upload_to_milvus(figures, tables)

                results[idx] = (figures, tables)
                total_figures += len(figures)
                total_tables += len(tables)

                self.logger.info(
                    f"[{idx+1}/{n}] {paper.id}: "
                    f"{len(figures)} figures, {len(tables)} tables"
                )
            except Exception as e:
                self.logger.error(
                    f"[{idx+1}/{n}] {paper.id}: extraction failed: {e}"
                )
            finally:
                try:
                    os.remove(pdf_path)
                except OSError:
                    pass

        self.logger.info(
            f"Phase 2 complete: {total_figures} figures, "
            f"{total_tables} tables from {len(converted)} papers"
        )
        return results

    def _upload_to_milvus(self, figures: List[Figure], tables: List[Table]) -> None:
        """Upload figures and tables to Zilliz/Milvus (best-effort)."""
        try:
            if figures:
                figures_success = self._save_figures_to_milvus(figures)
                if not figures_success:
                    self.logger.warning(
                        "Failed to save some figures to Milvus"
                    )
            if tables:
                tables_success = self._save_tables_to_milvus(tables)
                if not tables_success:
                    self.logger.warning(
                        "Failed to save some tables to Milvus"
                    )
        except Exception as e:
            self.logger.error(f"Error saving to Milvus: {e}")
