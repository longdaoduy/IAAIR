"""
PDF Processing Handler for extracting figures and tables from academic papers.

Uses pymupdf4llm for robust, LLM-optimised extraction of images and tables
from PDF documents.  pymupdf4llm handles multi-column layouts, colorspace
quirks and table detection out of the box.
"""

import hashlib
import logging
import os
import requests
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


class PDFProcessingHandler:
    """Handler for processing PDFs and extracting visual content."""

    MAX_FIGURES_PER_PAPER = 10   # cap extracted figures per PDF
    MAX_TABLES_PER_PAPER  = 10   # cap extracted tables per PDF

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

    def extract_figures_and_tables(self, pdf_path: str, paper_id: str) -> Tuple[List[Figure], List[Table]]:
        """
        Extract figures and tables from a PDF using pymupdf4llm.

        pymupdf4llm's ``to_markdown`` with ``page_chunks=True`` and
        ``write_images=True`` gives us:
          • Per-page markdown text (tables rendered as GFM markdown tables)
          • Image files saved to disk with references in the markdown
          • Structured metadata per page (images list, tables list)

        All CLIP/SciBERT embeddings are generated in batches at the end
        for maximum throughput.
        """
        figures: List[Figure] = []
        tables: List[Table] = []

        try:
            # Use a paper-specific subfolder so images don't collide
            paper_image_dir = os.path.join(self.figures_dir, paper_id)
            os.makedirs(paper_image_dir, exist_ok=True)

            # ── Run pymupdf4llm ──────────────────────────────────────────
            _common_kwargs = dict(
                page_chunks=True,
                write_images=True,
                image_path=paper_image_dir,
                image_format="png",
                dpi=150,                # 150 is sufficient and ~45% faster than 200
                image_size_limit=0.01,  # lowered from 0.05 – many sub-figures are <5% of page
                force_text=True,
                show_progress=False,
            )

            page_data = None
            for attempt, extra in enumerate([
                {},
                {"table_strategy": "lines"},
                {"table_strategy": "text"},
                {"ignore_graphics": True},
            ]):
                try:
                    page_data = _to_markdown(pdf_path, **{**_common_kwargs, **extra})
                    break
                except (ValueError, TypeError, AttributeError) as e:
                    self.logger.warning(
                        f"pymupdf4llm attempt {attempt + 1} failed for {paper_id}: {e}"
                    )
                    continue

            if page_data is None:
                self.logger.error(f"All pymupdf4llm attempts failed for {paper_id}")
                return figures, tables

            # ── Phase 1: Collect raw figure/table data (no embeddings yet) ──
            # We accumulate images and text descriptions so they can be
            # embedded in one batch call at the end.
            _FigureRaw = type('_FigureRaw', (), {})   # lightweight temp containers
            _TableRaw  = type('_TableRaw', (), {})

            raw_figures: List[object] = []   # list of _FigureRaw
            raw_tables:  List[object] = []   # list of _TableRaw

            figure_counter = 1
            table_counter = 1

            # Deduplication tracking
            _seen_hashes: Set[str] = set()        # exact MD5 of raw file bytes
            _seen_phashes: Set[str] = set()       # perceptual hash (8×8 dct)
            _skipped_dupes = 0

            for chunk in page_data:
                # Early exit: stop scanning pages once both limits are reached
                if (len(raw_figures) >= self.MAX_FIGURES_PER_PAPER
                        and len(raw_tables) >= self.MAX_TABLES_PER_PAPER):
                    break

                page_num = chunk["metadata"].get("page") or chunk["metadata"].get("page_number", 0)
                md_text = chunk.get("text", "")

                # ── 1. FIGURES ───────────────────────────────────────────
                if len(raw_figures) < self.MAX_FIGURES_PER_PAPER:
                    img_pattern = re.compile(
                        r'!\[([^\]]*)\]\(([^)]+\.(?:png|jpg|jpeg|webp|gif))\)',
                        re.IGNORECASE,
                    )
                    for match in img_pattern.finditer(md_text):
                        alt_text = match.group(1).strip()
                        img_rel_path = match.group(2).strip()

                        if not os.path.exists(img_rel_path):
                            self.logger.warning(f"Image file not found: {img_rel_path}")
                            continue

                        # ── Exact-hash dedup (fast, catches identical files) ──
                        try:
                            raw_bytes = open(img_rel_path, 'rb').read()
                            md5 = hashlib.md5(raw_bytes).hexdigest()
                        except Exception:
                            md5 = None
                        if md5 and md5 in _seen_hashes:
                            _skipped_dupes += 1
                            self.logger.debug(f"Skipped exact-duplicate image: {img_rel_path}")
                            try:
                                os.remove(img_rel_path)
                            except OSError:
                                pass
                            continue
                        if md5:
                            _seen_hashes.add(md5)

                        try:
                            pil_image = Image.open(img_rel_path).convert("RGB")
                        except Exception as e:
                            self.logger.warning(f"Cannot open image {img_rel_path}: {e}")
                            continue

                        if pil_image.width < 100 or pil_image.height < 100:
                            continue

                        # ── Perceptual-hash dedup (catches resized/recompressed dupes) ──
                        phash = self._compute_image_phash(pil_image)
                        if phash and phash in _seen_phashes:
                            _skipped_dupes += 1
                            self.logger.debug(f"Skipped perceptually-duplicate image: {img_rel_path}")
                            try:
                                os.remove(img_rel_path)
                            except OSError:
                                pass
                            continue
                        if phash:
                            _seen_phashes.add(phash)

                        # A figure is only valid if we can find a real
                        # caption near it that references "Figure" / "Fig".
                        # Pure alt-text from the markdown tag is NOT enough.
                        description = self._find_caption_near_image(
                            md_text, match.start(),
                            prefix_pattern=r'(?:fig(?:ure|\.)?)\s*\.?\s*(?:S?\d+[a-z]?(?:\([a-z]\))?)',
                            window=800,
                        )

                        if not description or not re.search(r'\bfig(?:ure)?\b', description, re.IGNORECASE):
                            # No valid caption → discard the image file
                            try:
                                os.remove(img_rel_path)
                            except OSError:
                                pass
                            continue

                        figure_id = f"{paper_id}#figure_{figure_counter}"
                        canonical_path = os.path.join(self.figures_dir, f"{figure_id}.png")
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
                        rf.pil_image = pil_image          # keep for batch CLIP later
                        raw_figures.append(rf)
                        figure_counter += 1
                        if len(raw_figures) >= self.MAX_FIGURES_PER_PAPER:
                            break

                # ── 2. TABLES ────────────────────────────────────────────
                if len(raw_tables) >= self.MAX_TABLES_PER_PAPER:
                    continue  # skip table extraction, limit reached

                table_blocks = self._extract_markdown_tables(md_text)
                for table_md in table_blocks:
                    table_caption = self._find_caption_near_table(
                        md_text, table_md,
                        prefix_pattern=r'(?:tab(?:le|\.)?)\s*\.?\s*(?:S?\d+[a-z]?)',
                        window=600,
                    )

                    headers, rows = self._parse_markdown_table(table_md)
                    if not headers and not rows:
                        continue

                    table_id = f"{paper_id}#table_{table_counter}"
                    description = table_caption or None

                    # Only keep tables whose caption references a table
                    if not description or not re.search(r'\btab(?:le)?\b', description, re.IGNORECASE):
                        continue

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
                    if len(raw_tables) >= self.MAX_TABLES_PER_PAPER:
                        break

            # ── Phase 1a-fallback: PyMuPDF native image extraction ──────
            # If pymupdf4llm didn't produce enough figures (e.g. vector
            # graphics, inline images it didn't export), fall back to
            # PyMuPDF's page.get_images() which extracts all raster images
            # embedded in the PDF XObjects.
            if len(raw_figures) < self.MAX_FIGURES_PER_PAPER:
                try:
                    doc = fitz.open(pdf_path)
                    full_text_by_page: Dict[int, str] = {}

                    for page_idx in range(doc.page_count):
                        if len(raw_figures) >= self.MAX_FIGURES_PER_PAPER:
                            break

                        page = doc[page_idx]
                        page_num = page_idx + 1

                        # Cache page text for caption search
                        if page_num not in full_text_by_page:
                            full_text_by_page[page_num] = page.get_text("text")

                        image_list = page.get_images(full=True)
                        for img_info in image_list:
                            if len(raw_figures) >= self.MAX_FIGURES_PER_PAPER:
                                break

                            xref = img_info[0]
                            try:
                                base_image = doc.extract_image(xref)
                                if not base_image or not base_image.get("image"):
                                    continue

                                img_bytes = base_image["image"]
                                pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                                # Skip small images (icons, logos, decorations)
                                if pil_image.width < 100 or pil_image.height < 100:
                                    continue

                                # Exact-hash dedup
                                md5 = hashlib.md5(img_bytes).hexdigest()
                                if md5 in _seen_hashes:
                                    _skipped_dupes += 1
                                    continue
                                _seen_hashes.add(md5)

                                # Perceptual-hash dedup
                                phash = self._compute_image_phash(pil_image)
                                if phash and phash in _seen_phashes:
                                    _skipped_dupes += 1
                                    continue
                                if phash:
                                    _seen_phashes.add(phash)

                                # Search for a figure caption on this page
                                page_text = full_text_by_page[page_num]
                                description = self._find_caption_in_page_text(
                                    page_text,
                                    prefix_pattern=r'(?:fig(?:ure|\.)?)\s*\.?\s*(?:S?\d+[a-z]?(?:\([a-z]\))?)',
                                )
                                if not description or not re.search(
                                    r'\bfig(?:ure)?\b', description, re.IGNORECASE
                                ):
                                    continue

                                figure_id = f"{paper_id}#figure_{figure_counter}"
                                canonical_path = os.path.join(
                                    self.figures_dir, f"{figure_id}.png"
                                )
                                pil_image.save(canonical_path, "PNG")

                                rf = _FigureRaw()
                                rf.figure_id = figure_id
                                rf.paper_id = paper_id
                                rf.figure_number = figure_counter
                                rf.description = description
                                rf.page_number = page_num
                                rf.image_path = canonical_path
                                rf.pil_image = pil_image
                                raw_figures.append(rf)
                                figure_counter += 1

                            except Exception as img_err:
                                self.logger.debug(
                                    f"Could not extract image xref={xref} "
                                    f"from page {page_num}: {img_err}"
                                )
                                continue

                    doc.close()
                    if figure_counter > 1:
                        self.logger.info(
                            f"PyMuPDF native fallback found {len(raw_figures)} "
                            f"total figures for {paper_id}"
                        )
                except Exception as e:
                    self.logger.warning(f"PyMuPDF native figure fallback failed: {e}")

            # ── Phase 1b: Remove sub-images contained in larger figures ─
            raw_figures, containment_removed = self._remove_contained_sub_images(raw_figures)
            _skipped_dupes += containment_removed

            if _skipped_dupes > 0:
                self.logger.info(
                    f"Dedup: skipped {_skipped_dupes} duplicate/sub-images for {paper_id}"
                )

            # ── Phase 1c: Cap figures & tables to configured limits ──────
            if len(raw_figures) > self.MAX_FIGURES_PER_PAPER:
                # Keep the largest images (by pixel area) — they are the
                # most likely to be full figures rather than icons/logos.
                raw_figures.sort(
                    key=lambda rf: rf.pil_image.width * rf.pil_image.height,
                    reverse=True,
                )
                excess = raw_figures[self.MAX_FIGURES_PER_PAPER:]
                raw_figures = raw_figures[:self.MAX_FIGURES_PER_PAPER]
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
                        new_path = os.path.join(
                            os.path.dirname(rf.image_path), f"{new_id}.png"
                        )
                        try:
                            if os.path.exists(rf.image_path):
                                os.rename(rf.image_path, new_path)
                            rf.image_path = new_path
                        except OSError:
                            pass
                        rf.figure_id = new_id
                    rf.figure_number = i
                self.logger.info(
                    f"Capped figures from {len(raw_figures) + len(excess)} "
                    f"to {self.MAX_FIGURES_PER_PAPER} for {paper_id}"
                )

            if len(raw_tables) > self.MAX_TABLES_PER_PAPER:
                raw_tables = raw_tables[:self.MAX_TABLES_PER_PAPER]
                for i, rt in enumerate(raw_tables, 1):
                    rt.table_id = f"{rt.paper_id}#table_{i}"
                    rt.table_number = i
                self.logger.info(
                    f"Capped tables to {self.MAX_TABLES_PER_PAPER} for {paper_id}"
                )

            # ── Phase 2: Render table images (open PDF only once) ────────
            table_images: List[Tuple[Optional[str], Optional[Image.Image]]] = []
            if raw_tables:
                try:
                    doc = fitz.open(pdf_path)
                    for rt in raw_tables:
                        img_path, pil_img = self._render_table_image_from_doc(
                            doc, rt.page_number - 1, rt.table_md, rt.table_id
                        )
                        table_images.append((img_path, pil_img))
                    doc.close()
                except Exception as e:
                    self.logger.error(f"Error rendering table images: {e}")
                    table_images = [(None, None)] * len(raw_tables)

            # ── Phase 3: Batch embeddings ────────────────────────────────
            # Collect all images and texts to embed in one call each.
            fig_pil_images   = [rf.pil_image for rf in raw_figures]
            fig_descriptions = [rf.description for rf in raw_figures]

            tbl_descriptions = [rt.description for rt in raw_tables]
            tbl_pil_images   = [ti[1] for ti in table_images]  # may contain None

            # CLIP image embeddings: figures + tables in one batch
            all_clip_images: List[Optional[Image.Image]] = fig_pil_images + tbl_pil_images
            if all_clip_images:
                all_clip_embs = self.clip_client.generate_image_embeddings_batch(
                    [img for img in all_clip_images],  # pass through, batch method handles None
                )
            else:
                all_clip_embs = []

            fig_clip_embs = all_clip_embs[:len(raw_figures)]
            tbl_clip_embs = all_clip_embs[len(raw_figures):]

            # SciBERT text embeddings: figure descriptions + table descriptions in one batch
            all_scibert_texts = fig_descriptions + tbl_descriptions
            if all_scibert_texts:
                all_scibert_embs = self.scibert_client.generate_text_embeddings_batch(all_scibert_texts)
            else:
                all_scibert_embs = []

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

            # Clean up leftover pymupdf4llm temp images that weren't
            # matched as figures (e.g. *.pdf-1-0.png in the paper subfolder)
            try:
                for leftover in os.listdir(paper_image_dir):
                    leftover_path = os.path.join(paper_image_dir, leftover)
                    if os.path.isfile(leftover_path):
                        os.remove(leftover_path)
                os.rmdir(paper_image_dir)
            except OSError:
                pass

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

        Uses a simplified DCT-based approach:
        1. Resize to (hash_size*4) × (hash_size*4) with antialiasing
        2. Convert to greyscale
        3. Compute a simple block-average hash (fast, no scipy needed)

        Returns a hex string, or None on error.
        """
        try:
            # Resize to a small square and convert to greyscale
            small = pil_image.resize((hash_size * 4, hash_size * 4), Image.LANCZOS).convert("L")
            pixels = list(small.getdata())
            avg = sum(pixels) / len(pixels)
            # Build a bit string: 1 if pixel > average, else 0
            bits = ''.join('1' if p > avg else '0' for p in pixels)
            # Convert to hex for compact storage
            hash_int = int(bits, 2)
            return format(hash_int, f'0{len(bits) // 4}x')
        except Exception:
            return None

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
        from the same page.

        Two images on the same page where the smaller one's area is
        < area_ratio_threshold of the larger, and their perceptual hashes
        are within a tolerance, means the smaller is likely a sub-panel.

        Also removes images whose pHash is nearly identical (distance ≤ 5)
        regardless of size — those are true duplicates.

        Returns (filtered_figures, num_removed).
        """
        if len(raw_figures) <= 1:
            return raw_figures, 0

        # Group by page
        by_page: Dict[int, List[object]] = {}
        for rf in raw_figures:
            by_page.setdefault(rf.page_number, []).append(rf)

        keep_ids: Set[str] = set()
        removed = 0

        for page_num, page_figs in by_page.items():
            if len(page_figs) <= 1:
                for rf in page_figs:
                    keep_ids.add(rf.figure_id)
                continue

            # Compute area and phash for each figure on this page
            areas = []
            phashes = []
            for rf in page_figs:
                w, h = rf.pil_image.size
                areas.append(w * h)
                phashes.append(self._compute_image_phash(rf.pil_image))

            # Sort by area descending so we compare smaller against larger
            indexed = sorted(range(len(page_figs)), key=lambda i: areas[i], reverse=True)
            removed_on_page: Set[int] = set()

            for i_pos, i in enumerate(indexed):
                if i in removed_on_page:
                    continue
                for j in indexed[i_pos + 1:]:
                    if j in removed_on_page:
                        continue

                    # Near-identical pHash → duplicate regardless of size
                    if phashes[i] and phashes[j]:
                        dist = self._phash_distance(phashes[i], phashes[j])
                        if dist <= 5:
                            # Keep the larger one
                            removed_on_page.add(j)
                            self.logger.debug(
                                f"Dedup: {page_figs[j].figure_id} is near-identical "
                                f"to {page_figs[i].figure_id} (pHash dist={dist})"
                            )
                            continue

                    # Sub-image containment: small image << large image
                    if areas[i] > 0:
                        ratio = areas[j] / areas[i]
                        if ratio < area_ratio_threshold and phashes[i] and phashes[j]:
                            # Looser pHash threshold for containment
                            dist = self._phash_distance(phashes[i], phashes[j])
                            if dist <= 20:
                                removed_on_page.add(j)
                                self.logger.debug(
                                    f"Dedup: {page_figs[j].figure_id} is a sub-image "
                                    f"of {page_figs[i].figure_id} "
                                    f"(area ratio={ratio:.2f}, pHash dist={dist})"
                                )

            for idx in range(len(page_figs)):
                if idx in removed_on_page:
                    removed += 1
                    # Clean up the file
                    try:
                        os.remove(page_figs[idx].image_path)
                    except OSError:
                        pass
                else:
                    keep_ids.add(page_figs[idx].figure_id)

        # Re-number figures sequentially after removing duplicates
        filtered = [rf for rf in raw_figures if rf.figure_id in keep_ids]
        for i, rf in enumerate(filtered, 1):
            old_id = rf.figure_id
            new_id = f"{rf.paper_id}#figure_{i}"
            if old_id != new_id:
                new_path = os.path.join(
                    os.path.dirname(rf.image_path),
                    f"{new_id}.png"
                )
                try:
                    if os.path.exists(rf.image_path):
                        os.rename(rf.image_path, new_path)
                    rf.image_path = new_path
                except OSError:
                    pass
                rf.figure_id = new_id
            rf.figure_number = i

        return filtered, removed

    # ─── Helper methods for the pymupdf4llm-based extraction ────────────

    @staticmethod
    def _find_caption_near_image(md_text: str, img_pos: int,
                                 prefix_pattern: str, window: int = 800) -> Optional[str]:
        """
        Search for a caption (e.g. "Figure 1: …") in the markdown text
        within *window* characters before/after the image reference.

        Uses multiple strategies:
        1. Standard prefix match ("Figure 1: description…")
        2. Bold/italic caption ("**Figure 1.** description…")
        3. Parenthetical label ((a) description) near the image
        """
        start = max(0, img_pos - window)
        end = min(len(md_text), img_pos + window)
        neighbourhood = md_text[start:end]

        # Strategy 1: Standard caption – "Figure 1: …" or "Fig. 2. …"
        # Terminate at double-newline or a line that starts a new section
        # (heading, another figure/table, or a pipe-table row).
        pattern = re.compile(
            rf'({prefix_pattern}[.:\-—]?\s*.+?)'
            r'(?:\n\n|\n(?=#{1,3}\s)|\n(?=(?:fig(?:ure|\.)?|tab(?:le|\.)?|\|)\s*\.?\s*\d)|$)',
            re.IGNORECASE | re.DOTALL,
        )
        m = pattern.search(neighbourhood)
        if m:
            caption = m.group(1).strip()
            # Trim excessively long captions (> 1000 chars) – likely runaway match
            if len(caption) > 1000:
                caption = caption[:1000].rsplit('.', 1)[0] + '.'
            return caption

        # Strategy 2: Bold / italic markdown caption
        # e.g. "**Figure 1.** Some description here"
        bold_pattern = re.compile(
            r'(\*{1,2}(?:fig(?:ure|\.)?\s*\.?\s*S?\d+[a-z]?)\*{1,2}[.:\-—]?\s*.+?)'
            r'(?:\n\n|\n(?=#{1,3}\s)|$)',
            re.IGNORECASE | re.DOTALL,
        )
        m2 = bold_pattern.search(neighbourhood)
        if m2:
            caption = m2.group(1).strip().strip('*').strip()
            if len(caption) > 1000:
                caption = caption[:1000].rsplit('.', 1)[0] + '.'
            return caption

        return None

    @staticmethod
    def _find_caption_in_page_text(page_text: str,
                                   prefix_pattern: str) -> Optional[str]:
        """
        Search raw page text (from PyMuPDF ``page.get_text("text")``) for
        a figure/table caption.  Used by the native-extraction fallback
        where we don't have a specific image position in the markdown.

        Returns the first matching caption, or None.
        """
        if not page_text:
            return None

        # Standard caption line(s)
        pattern = re.compile(
            rf'({prefix_pattern}[.:\-—]?\s*.+?)'
            r'(?:\n\n|\n(?=(?:fig(?:ure|\.)?|tab(?:le|\.)?)\s*\.?\s*\d)|$)',
            re.IGNORECASE | re.DOTALL,
        )
        m = pattern.search(page_text)
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

        Checks performed:
        1. Must have a proper separator row (cells like ``---``, ``:---:``, etc.)
        2. Separator row must be the 2nd line (header | separator | data…)
        3. Must have at least 2 columns
        4. Must have at least 1 data row (below the separator)
        5. Column count must be roughly consistent across rows
        6. Data cells must not all be empty / whitespace
        7. Reject if most cells look like long prose (>120 chars avg)
        """
        if len(table_lines) < 3:          # header + separator + ≥1 data row
            return False

        def _split_cells(line: str) -> List[str]:
            return [c.strip() for c in line.strip('|').split('|')]

        # ── Locate separator row (must be line index 1) ──────────────
        sep_line = table_lines[1]
        sep_cells = _split_cells(sep_line)

        # Each separator cell must be composed of dashes, colons, spaces only
        # and must contain at least 3 dashes  (e.g. ``---``, ``:---:``)
        SEP_CELL = re.compile(r'^:?\s*-{3,}\s*:?$')
        if not all(SEP_CELL.match(c) for c in sep_cells):
            return False

        num_cols = len(sep_cells)
        if num_cols < 2:
            return False

        # ── Header column count must match separator ─────────────────
        header_cells = _split_cells(table_lines[0])
        if abs(len(header_cells) - num_cols) > 1:
            return False

        # ── Must have at least 1 data row ────────────────────────────
        data_lines = table_lines[2:]
        if not data_lines:
            return False

        # ── Column-count consistency: allow ±1 difference ────────────
        mismatches = 0
        for dl in data_lines:
            row_cols = len(_split_cells(dl))
            if abs(row_cols - num_cols) > 1:
                mismatches += 1
        if mismatches > len(data_lines) * 0.5:
            return False

        # ── Data rows must not be all-empty ──────────────────────────
        non_empty_cells = 0
        total_cell_len = 0
        total_cells = 0
        for dl in data_lines:
            for c in _split_cells(dl):
                total_cells += 1
                total_cell_len += len(c)
                if c:
                    non_empty_cells += 1
        if non_empty_cells == 0:
            return False

        # ── Reject prose masquerading as a table ─────────────────────
        # If the average cell length is very long, it's likely a
        # multi-column layout or bibliography, not a data table.
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
                                 prefix_pattern: str, window: int = 600) -> Optional[str]:
        """
        Find a table caption in the neighbourhood of the table block.

        Searches both above and below the table for lines starting
        with "Table N" / "Tab. N" etc.
        """
        pos = md_text.find(table_md)
        if pos == -1:
            return None
        start = max(0, pos - window)
        end = min(len(md_text), pos + len(table_md) + window)
        neighbourhood = md_text[start:end]

        # Strategy 1: standard caption line
        pattern = re.compile(
            rf'({prefix_pattern}[.:\-—]?\s*.+?)'
            r'(?:\n\n|\n(?=\|)|\n(?=#{1,3}\s)|\n(?=(?:fig(?:ure|\.)?|tab(?:le|\.)?)\.?\s*\d)|$)',
            re.IGNORECASE | re.DOTALL,
        )
        m = pattern.search(neighbourhood)
        if m:
            caption = m.group(1).strip()
            if len(caption) > 1000:
                caption = caption[:1000].rsplit('.', 1)[0] + '.'
            return caption

        # Strategy 2: bold/italic table caption
        bold_pattern = re.compile(
            r'(\*{1,2}(?:tab(?:le|\.)?\s*\.?\s*S?\d+[a-z]?)\*{1,2}[.:\-—]?\s*.+?)'
            r'(?:\n\n|\n(?=\|)|$)',
            re.IGNORECASE | re.DOTALL,
        )
        m2 = bold_pattern.search(neighbourhood)
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
            if re.match(r'^[\s|:-]+$', line):
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

    def process_paper_pdf(self, paper: Paper) -> Tuple[List[Figure], List[Table]]:
        """
        Complete pipeline to process a paper's PDF and extract visual content.
        
        Args:
            paper: Paper entity with pdf_url
            
        Returns:
            Tuple of (figures, tables) extracted from the PDF
        """
        if not paper.pdf_url:
            self.logger.warning(f"No PDF URL provided for paper {paper.id}")
            return [], []

        # Download PDF
        pdf_path = self.download_pdf(paper.pdf_url, paper.id)
        if not pdf_path:
            return [], []

        # Extract figures and tables with embeddings
        figures, tables = self.extract_figures_and_tables(pdf_path, paper.id)

        # Save embeddings to Milvus using dedicated collections
        try:
            self.logger.info(f"Saving {len(figures)} figures and {len(tables)} tables to Milvus...")

            # Save figures to dedicated figures collection
            figures_success = self._save_figures_to_milvus(figures)
            if not figures_success:
                self.logger.warning("Failed to save some figures to dedicated Milvus figures collection")

            # Save tables to dedicated tables collection
            tables_success = self._save_tables_to_milvus(tables)
            if not tables_success:
                self.logger.warning("Failed to save some tables to dedicated Milvus tables collection")

            if figures_success and tables_success:
                self.logger.info("Successfully saved all figures and tables to their dedicated Milvus collections")

        except Exception as e:
            self.logger.error(f"Error saving to Milvus: {e}")

        # Clean up PDF file (optional)
        try:
            os.remove(pdf_path)
        except:
            pass

        return figures, tables
