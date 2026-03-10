"""
PDF Processing Handler for extracting figures and tables from academic papers.

Uses pymupdf4llm for robust, LLM-optimised extraction of images and tables
from PDF documents.  pymupdf4llm handles multi-column layouts, colorspace
quirks and table detection out of the box.
"""

import logging
import os
import requests
from typing import List, Optional, Tuple
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
                image_size_limit=0.05,
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

            for chunk in page_data:
                page_num = chunk["metadata"].get("page") or chunk["metadata"].get("page_number", 0)
                md_text = chunk.get("text", "")

                # ── 1. FIGURES ───────────────────────────────────────────
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

                    try:
                        pil_image = Image.open(img_rel_path).convert("RGB")
                    except Exception as e:
                        self.logger.warning(f"Cannot open image {img_rel_path}: {e}")
                        continue

                    if pil_image.width < 100 or pil_image.height < 100:
                        continue

                    description = self._find_caption_near_image(
                        md_text, match.start(), prefix_pattern=r'(?:fig(?:ure)?)\s*\.?\s*\d+'
                    ) or alt_text or None

                    if not description:
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

                # ── 2. TABLES ────────────────────────────────────────────
                table_blocks = self._extract_markdown_tables(md_text)
                for table_md in table_blocks:
                    table_caption = self._find_caption_near_table(
                        md_text, table_md,
                        prefix_pattern=r'(?:table|tab)\s*\.?\s*\d+'
                    )

                    headers, rows = self._parse_markdown_table(table_md)
                    if not headers and not rows:
                        continue

                    table_id = f"{paper_id}#table_{table_counter}"
                    description = table_caption or table_md[:8000]

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

            self.logger.info(
                f"Extracted {len(figures)} figures and {len(tables)} tables "
                f"from {paper_id} using pymupdf4llm (batch embeddings)"
            )

        except Exception as e:
            self.logger.error(f"Critical error processing PDF {pdf_path}: {e}", exc_info=True)

        return figures, tables

    # ─── Helper methods for the pymupdf4llm-based extraction ────────────

    @staticmethod
    def _find_caption_near_image(md_text: str, img_pos: int,
                                 prefix_pattern: str, window: int = 600) -> Optional[str]:
        """
        Search for a caption (e.g. "Figure 1: …") in the markdown text
        within *window* characters before/after the image reference.
        """
        start = max(0, img_pos - window)
        end = min(len(md_text), img_pos + window)
        neighbourhood = md_text[start:end]

        # Match lines that start with "Figure N" / "Fig. N"
        pattern = re.compile(
            rf'({prefix_pattern}[.:]?\s*.+?)(?:\n\n|\n(?=[#|\[])|$)',
            re.IGNORECASE | re.DOTALL,
        )
        m = pattern.search(neighbourhood)
        if m:
            return m.group(1).strip()
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
                                 prefix_pattern: str, window: int = 400) -> Optional[str]:
        """Find a table caption in the neighbourhood of the table block."""
        pos = md_text.find(table_md)
        if pos == -1:
            return None
        start = max(0, pos - window)
        end = min(len(md_text), pos + len(table_md) + window)
        neighbourhood = md_text[start:end]

        pattern = re.compile(
            rf'({prefix_pattern}[.:]?\s*.+?)(?:\n\n|\n(?=\|)|$)',
            re.IGNORECASE | re.DOTALL,
        )
        m = pattern.search(neighbourhood)
        if m:
            return m.group(1).strip()
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

        Returns (image_path, pil_image) — the PIL image is kept so the
        caller can pass it into a batch CLIP embedding call later.
        """
        try:
            if page_index >= doc.page_count or page_index < 0:
                return None, None

            page = doc[page_index]

            lines = [l.strip() for l in table_md.strip().split('\n') if l.strip()]
            if not lines:
                return None, None

            header_cells = [c.strip() for c in lines[0].strip('|').split('|') if c.strip()]
            search_term = header_cells[0] if header_cells else None

            table_rect = None
            if search_term:
                instances = page.search_for(search_term)
                if instances:
                    table_rect = instances[0]
                    if len(header_cells) > 1:
                        last_instances = page.search_for(header_cells[-1])
                        if last_instances:
                            table_rect = table_rect | last_instances[0]

                    num_data_rows = len([l for l in lines[1:] if not re.match(r'^[\s|:-]+$', l)])
                    table_rect.y1 = min(page.rect.height, table_rect.y1 + num_data_rows * 22)

                    padding = 12
                    table_rect.x0 = max(0, table_rect.x0 - padding)
                    table_rect.y0 = max(0, table_rect.y0 - padding)
                    table_rect.x1 = min(page.rect.width, table_rect.x1 + padding)
                    table_rect.y1 = min(page.rect.height, table_rect.y1 + padding)

            if table_rect is None or table_rect.is_empty:
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
        Render the table region on the given PDF page as a PNG image and
        generate a CLIP embedding for it.

        We search the page text for the first cell of the table header to
        locate the table's bounding box, then clip and render that region.
        """
        try:
            doc = fitz.open(pdf_path)
            if page_index >= doc.page_count:
                doc.close()
                return None, None

            page = doc[page_index]

            # Try to find table bbox by searching for header cells on the page
            lines = [l.strip() for l in table_md.strip().split('\n') if l.strip()]
            if not lines:
                doc.close()
                return None, None

            header_cells = [c.strip() for c in lines[0].strip('|').split('|') if c.strip()]
            search_term = header_cells[0] if header_cells else None

            table_rect = None
            if search_term:
                instances = page.search_for(search_term)
                if instances:
                    # Start with the first hit and expand with other cells
                    table_rect = instances[0]
                    # Also search for last header cell to get width
                    if len(header_cells) > 1:
                        last_instances = page.search_for(header_cells[-1])
                        if last_instances:
                            table_rect = table_rect | last_instances[0]

                    # Expand downward to cover all rows  (estimate ~20pt per row)
                    num_data_rows = len([l for l in lines[1:] if not re.match(r'^[\s|:-]+$', l)])
                    table_rect.y1 = min(page.rect.height, table_rect.y1 + num_data_rows * 22)

                    # Add padding
                    padding = 12
                    table_rect.x0 = max(0, table_rect.x0 - padding)
                    table_rect.y0 = max(0, table_rect.y0 - padding)
                    table_rect.x1 = min(page.rect.width, table_rect.x1 + padding)
                    table_rect.y1 = min(page.rect.height, table_rect.y1 + padding)

            if table_rect is None or table_rect.is_empty:
                # Fallback: render the whole page
                table_rect = page.rect

            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat, clip=table_rect)

            # Convert to PIL
            img_data = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_data)).convert("RGB")

            image_path = os.path.join(self.tables_dir, f"{table_id}_image.png")
            pil_image.save(image_path)

            image_embedding = None
            try:
                image_embedding = self.clip_client.generate_image_embedding(pil_image)
            except Exception as embed_err:
                self.logger.warning(f"Failed embedding for table {table_id}: {embed_err}")

            pix = None
            doc.close()
            self.logger.info(f"Rendered table image for {table_id}: {image_path}")
            return image_path, image_embedding

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
