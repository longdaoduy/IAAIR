"""
PDF Processing Handler for extracting figures and tables from academic papers.

Uses pymupdf4llm for robust, LLM-optimised extraction of images and tables
from PDF documents.  pymupdf4llm handles multi-column layouts, colorspace
quirks and table detection out of the box.
"""

import logging
import os
import json
import requests
from typing import List, Optional, Tuple
import fitz  # PyMuPDF
import pymupdf4llm
from PIL import Image
import io
import re
from datetime import datetime

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

        We post-process that output to build our Figure / Table entities
        with CLIP and SciBERT embeddings.
        """
        figures: List[Figure] = []
        tables: List[Table] = []

        try:
            # Use a paper-specific subfolder so images don't collide
            paper_image_dir = os.path.join(self.figures_dir, paper_id)
            os.makedirs(paper_image_dir, exist_ok=True)

            # ── Run pymupdf4llm ──────────────────────────────────────────
            page_data = pymupdf4llm.to_markdown(
                pdf_path,
                page_chunks=True,       # one dict per page
                write_images=True,      # save images to disk
                image_path=paper_image_dir,
                image_format="png",
                dpi=200,                # good balance quality/speed
                image_size_limit=0.05,  # skip tiny icons (<5% of page)
                force_text=True,        # also keep text overlapping images
                show_progress=False,
            )

            figure_counter = 1
            table_counter = 1

            for chunk in page_data:
                page_num = chunk["metadata"]["page_number"]  # 1-based
                md_text = chunk.get("text", "")

                # ── 1. FIGURES ───────────────────────────────────────────
                # pymupdf4llm writes images and inserts markdown references
                # like  ![<desc>](path/to/image.png)
                img_pattern = re.compile(
                    r'!\[([^\]]*)\]\(([^)]+\.(?:png|jpg|jpeg|webp|gif))\)',
                    re.IGNORECASE
                )
                for match in img_pattern.finditer(md_text):
                    alt_text = match.group(1).strip()
                    img_rel_path = match.group(2).strip()

                    # Resolve to absolute path
                    if not os.path.isabs(img_rel_path):
                        img_abs_path = os.path.normpath(
                            os.path.join(os.path.dirname(pdf_path), img_rel_path)
                        )
                    else:
                        img_abs_path = img_rel_path

                    if not os.path.exists(img_abs_path):
                        self.logger.warning(f"Image file not found: {img_abs_path}")
                        continue

                    try:
                        pil_image = Image.open(img_abs_path).convert("RGB")
                    except Exception as e:
                        self.logger.warning(f"Cannot open image {img_abs_path}: {e}")
                        continue

                    # Filter tiny images that slipped through
                    if pil_image.width < 100 or pil_image.height < 100:
                        continue

                    # Look for a "Figure N" / "Fig. N" caption nearby in the markdown
                    description = self._find_caption_near_image(
                        md_text, match.start(), prefix_pattern=r'(?:fig(?:ure)?)\s*\.?\s*\d+'
                    ) or alt_text or None

                    if not description:
                        continue

                    # Generate embeddings
                    image_embedding = self.clip_client.generate_image_embedding(pil_image)
                    desc_embedding = self.scibert_client.generate_text_embedding(description)

                    # Rename saved image to our canonical path
                    figure_id = f"{paper_id}#figure_{figure_counter}"
                    canonical_path = os.path.join(self.figures_dir, f"{figure_id}.png")
                    try:
                        if img_abs_path != canonical_path:
                            os.rename(img_abs_path, canonical_path)
                    except OSError:
                        canonical_path = img_abs_path  # keep original

                    figures.append(Figure(
                        id=figure_id,
                        paper_id=paper_id,
                        figure_number=figure_counter,
                        description=description,
                        page_number=page_num,
                        image_path=canonical_path,
                        image_embedding=image_embedding,
                        description_embedding=desc_embedding,
                    ))
                    figure_counter += 1

                # ── 2. TABLES ────────────────────────────────────────────
                # pymupdf4llm renders tables as GFM markdown tables.
                # We detect them with the standard  | … | pattern.
                table_blocks = self._extract_markdown_tables(md_text)
                for table_md in table_blocks:
                    # Try to find a "Table N" caption near the table block
                    table_caption = self._find_caption_near_table(
                        md_text, table_md,
                        prefix_pattern=r'(?:table|tab)\s*\.?\s*\d+'
                    )

                    headers, rows = self._parse_markdown_table(table_md)
                    if not headers and not rows:
                        continue

                    table_id = f"{paper_id}#table_{table_counter}"
                    description = table_caption or table_md[:8000]

                    # Embeddings
                    description_embedding = self.scibert_client.generate_text_embedding(description)

                    # Render the table region as an image for CLIP embedding
                    image_path, image_embedding = self._render_table_image_from_page(
                        pdf_path, page_num - 1, table_md, table_id
                    )

                    tables.append(Table(
                        id=table_id,
                        paper_id=paper_id,
                        table_number=table_counter,
                        description=description,
                        page_number=page_num,
                        headers=headers,
                        rows=rows,
                        description_embedding=description_embedding,
                        image_embedding=image_embedding,
                    ))
                    table_counter += 1

            self.logger.info(
                f"Extracted {len(figures)} figures and {len(tables)} tables "
                f"from {paper_id} using pymupdf4llm"
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
    def _extract_markdown_tables(md_text: str) -> List[str]:
        """
        Extract all GFM markdown table blocks from text.
        A markdown table is a sequence of lines starting/ending with ``|``
        that includes a separator row like ``|---|---|``.
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
                    # Validate: must contain a separator row  |---|...|
                    table_text = '\n'.join(current_table_lines)
                    if re.search(r'\|[\s-:]+\|', table_text):
                        tables.append(table_text)
                    current_table_lines = []
                in_table = False

        # Handle table at end of text
        if current_table_lines:
            table_text = '\n'.join(current_table_lines)
            if re.search(r'\|[\s-:]+\|', table_text):
                tables.append(table_text)

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

        Example input::

            | Model | Acc | F1 |
            |-------|-----|-----|
            | BERT  | 0.9 | 0.8 |
        """
        lines = [l.strip() for l in table_md.strip().split('\n') if l.strip()]
        if len(lines) < 2:
            return None, None

        def _split_row(line: str) -> List[str]:
            # Remove leading/trailing pipes and split
            return [c.strip() for c in line.strip('|').split('|')]

        headers = _split_row(lines[0])

        rows = []
        for line in lines[1:]:
            # Skip separator rows
            if re.match(r'^[\s|:-]+$', line):
                continue
            rows.append(_split_row(line))

        return headers, rows if rows else None

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
