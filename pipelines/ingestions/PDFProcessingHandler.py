"""
PDF Processing Handler for extracting figures and tables from academic papers.

This handler downloads PDFs, extracts visual content (figures and tables),
and generates descriptions and embeddings using various AI models.
"""

import logging
import os
import json
import requests
from typing import List, Optional, Tuple
import fitz  # PyMuPDF
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
        Extract figures and tables from PDF with robust colorspace handling
        and memory management.
        """
        figures = []
        tables = []

        try:
            doc = fitz.open(pdf_path)
            figure_counter = 1
            table_counter = 1

            for page_num in range(doc.page_count):
                page = doc[page_num]

                # --- 1. Figure Extraction ---
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    pix = None
                    try:
                        xref = img[0]
                        # PRIMARY FIX: Wrap the initial Pixmap creation
                        try:
                            pix = fitz.Pixmap(doc, xref)
                        except Exception:
                            # If colorspace is missing/unsupported, create an RGB Pixmap from the xref directly
                            pix = fitz.Pixmap(fitz.csRGB, doc, xref)

                        # Fix: Force colorspace conversion to RGB
                        # This solves the "unsupported colorspace for png" error
                        if pix.colorspace is None or pix.colorspace.n != 3:
                            pix = fitz.Pixmap(fitz.csRGB, pix)

                        # Fix: Use PPM as intermediate buffer for PIL
                        # PPM is a raw format that doesn't care about complex PDF colorspace metadata
                        try:
                            img_data = pix.tobytes("ppm")
                            pil_image = Image.open(io.BytesIO(img_data)).convert("RGB")
                        except Exception as conv_error:
                            self.logger.warning(f"Conversion failed on page {page_num + 1}: {conv_error}")
                            continue

                        # Filter: Ignore icons/tiny UI elements
                        if pil_image.width >= 100 and pil_image.height >= 100:
                            figure_id = f"{paper_id}#figure_{figure_counter}"
                            image_path = os.path.join(self.figures_dir, f"{figure_id}.png")

                            # Save and Generate Embeddings
                            pil_image.save(image_path)
                            image_embedding = self.clip_client.generate_image_embedding(pil_image)

                            description = self._extract_figure_description(page, img, page_num + 1)
                            if description:
                                desc_embedding = self.scibert_client.generate_embedding(description)

                                figures.append(Figure(
                                    id=figure_id,
                                    paper_id=paper_id,
                                    figure_number=figure_counter,
                                    description=description or "",
                                    page_number=page_num + 1,
                                    image_path=image_path,
                                    image_embedding=image_embedding,
                                    description_embedding=desc_embedding
                                ))
                                figure_counter += 1

                    except Exception as img_e:
                        self.logger.error(f"Image error on page {page_num + 1}: {img_e}")
                    finally:
                        # Explicitly release memory (important for 140+ page docs)
                        pix = None

                # --- 2. Table Extraction ---
                try:
                    page_tables = self._extract_tables_from_page(page, paper_id, table_counter, page_num + 1)
                    tables.extend(page_tables)
                    table_counter += len(page_tables)
                except Exception as tab_e:
                    self.logger.error(f"Table error on page {page_num + 1}: {tab_e}")

            doc.close()
            self.logger.info(f"Extracted {len(figures)} figures and {len(tables)} tables from {paper_id}")

        except Exception as e:
            self.logger.error(f"Critical error processing PDF {pdf_path}: {e}")

        return figures, tables

    def _extract_figure_description(self, page, img_info, page_num: int) -> Optional[str]:
        """
        Extract figure caption by analyzing proximity and vertical alignment.
        """
        try:
            # 1. Get image bounding box
            img_rects = page.get_image_rects(img_info[0])
            if not img_rects:
                return None
            img_rect = img_rects[0]

            # 2. Extract page text as dictionary for structural analysis
            blocks = page.get_text("dict")["blocks"]
            potential_captions = []

            for block in blocks:
                if "lines" not in block:
                    continue

                # Combine lines to handle multi-line captions
                block_text = ""
                for line in block["lines"]:
                    line_text = "".join([span["text"] for span in line["spans"]])
                    block_text += " " + line_text

                block_text = block_text.strip()
                block_rect = fitz.Rect(block["bbox"])

                # 3. Look for Figure/Fig prefix at the start of blocks
                # This distinguishes actual captions from "As shown in Fig 1..."
                if re.search(r'^(fig|figure)\.?\s*\d+', block_text, re.IGNORECASE):
                    # Calculate Euclidean distance
                    distance = self._calculate_distance(img_rect, block_rect)

                    # Heuristic: Captions are usually vertically aligned and close
                    # We give a "bonus" to blocks starting below the image
                    is_below = block_rect.y0 >= img_rect.y1 - 5

                    # Weighting: favor blocks below and very close
                    score = distance if is_below else distance * 2.0

                    potential_captions.append((block_text, score))

            if potential_captions:
                # Sort by our weighted score
                potential_captions.sort(key=lambda x: x[1])
                return potential_captions[0][0]

            return None

        except Exception as e:
            self.logger.error(f"Error extracting figure description: {e}")
            return None

    def _extract_tables_from_page(self, page, paper_id: str, start_counter: int, page_num: int) -> List[Table]:
        tables = []
        try:
            text_dict = page.get_text("dict")

            for block in text_dict["blocks"]:
                if "lines" not in block:
                    continue

                # Reconstruct block text
                block_text = ""
                for line in block["lines"]:
                    line_text = "".join([span["text"] for span in line["spans"]])
                    block_text += line_text + "\n"

                # Check if this block is the table body or contains the "Table X:" identifier
                is_table_content = self._is_likely_table(block_text)
                is_table_caption = re.search(r'^\s*(Table|Tab)\.?\s*\d+', block_text, re.IGNORECASE)

                if is_table_content and is_table_caption:
                    table_id = f"{paper_id}#table_{start_counter + len(tables)}"

                    # Extract surrounding description specifically looking for "Table X:"
                    description = self._extract_table_description(page, block["bbox"])

                    # If we found table-like text, generate embeddings
                    description_embedding = None
                    if block_text.strip():
                        # We embed the actual content of the table for semantic search
                        description_embedding = self.scibert_client.generate_embedding(block_text)

                    headers, rows = self._parse_table_structure(block_text)
                    image_path, image_embedding = self._capture_table_image(page, block["bbox"], table_id)

                    table = Table(
                        id=table_id,
                        paper_id=paper_id,
                        table_number=start_counter + len(tables),
                        description=description or block_text[:8000],  # Fallback to start of text
                        page_number=page_num,
                        headers=headers,
                        rows=rows,
                        description_embedding=description_embedding,
                        image_embedding=image_embedding
                    )
                    tables.append(table)

        except Exception as e:
            self.logger.error(f"Error extracting tables from page {page_num}: {e}")

        return tables

    def _is_likely_table(self, text: str) -> bool:
        """
        Heuristic to determine if text block is likely a table.
        
        Args:
            text: Text block to analyze
            
        Returns:
            True if likely a table
        """
        lines = text.strip().split('\n')

        # Simple heuristics
        if len(lines) < 3:  # Tables usually have at least 3 lines
            return False

        # Check for consistent column-like structure
        line_lengths = [len(line.split()) for line in lines if line.strip()]

        if not line_lengths:
            return False

        # Look for numerical data patterns
        numerical_lines = sum(1 for line in lines if re.search(r'\d+', line))

        # If more than half the lines contain numbers, likely a table
        return numerical_lines / len(lines) > 0.5

    def _extract_table_description(self, page, table_bbox) -> Optional[str]:
        """Extract caption for a table."""
        try:
            text_instances = page.get_text("dict")
            potential_captions = []

            for block in text_instances["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        line_rect = fitz.Rect(line["bbox"])

                        for span in line["spans"]:
                            line_text += span["text"]

                        line_text = line_text.strip()

                        # Check if text contains table references
                        if re.search(r'(table|tab)\s*\d+', line_text.lower()):
                            distance = self._calculate_distance(fitz.Rect(table_bbox), line_rect)
                            potential_captions.append((line_text, distance))

            if potential_captions:
                potential_captions.sort(key=lambda x: x[1])
                return potential_captions[0][0]

            return None

        except Exception as e:
            self.logger.error(f"Error extracting table description: {e}")
            return None

    def _parse_table_structure(self, table_text: str) -> Tuple[Optional[List[str]], Optional[List[List[str]]]]:
        """
        Parse table structure from text.
        
        Args:
            table_text: Raw table text
            
        Returns:
            Tuple of (headers, rows)
        """
        try:
            lines = [line.strip() for line in table_text.split('\n') if line.strip()]

            if len(lines) < 2:
                return None, None

            # Assume first line is headers
            headers = lines[0].split()

            # Parse remaining lines as rows
            rows = []
            for line in lines[1:]:
                row = line.split()
                if row:  # Only add non-empty rows
                    rows.append(row)

            return headers, rows if rows else None

        except Exception as e:
            self.logger.error(f"Error parsing table structure: {e}")
            return None, None

    def _calculate_distance(self, rect1: fitz.Rect, rect2: fitz.Rect) -> float:
        """Calculate distance between two rectangles."""
        center1 = ((rect1.x0 + rect1.x1) / 2, (rect1.y0 + rect1.y1) / 2)
        center2 = ((rect2.x0 + rect2.x1) / 2, (rect2.y0 + rect2.y1) / 2)

        return ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5

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
            # Connect to Milvus if not connected
            if not self.milvus_client.is_connected and not self.milvus_client.connect():
                self.logger.error("Failed to connect to Milvus")
                return False

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
            # Connect to Milvus if not connected
            if not self.milvus_client.is_connected and not self.milvus_client.connect():
                self.logger.error("Failed to connect to Milvus")
                return False

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
                    # Generate sparse embedding for combined description and table text
                    sparse_description_embedding = {0: 0.001}  # Default minimal sparse vector
                    # combined_text = f"{table.description or ''} {table.table_text or ''}".strip()
                    if table.description and self.milvus_client.is_tfidf_fitted:
                        sparse_description_embedding = self.milvus_client.generate_sparse_embedding(table.description)

                    # Ensure embedding is a valid list, not None
                    description_embedding = table.description_embedding if table.description_embedding is not None else [0.0] * 768

                    table_data = {
                        'id': table.id,
                        'paper_id': table.paper_id,
                        'description': table.description or "",
                        'description_embedding': description_embedding,
                        'sparse_description_embedding': sparse_description_embedding
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

    def _capture_table_image(self, page, bbox, table_id: str) -> Tuple[Optional[str], Optional[List[float]]]:
        """
        Capture table region as image and generate CLIP embedding.

        Args:
            page: PyMuPDF page object
            bbox: Bounding box of the table region
            table_id: Table ID for filename

        Returns:
            Tuple of (image_path, image_embedding) or (None, None) if failed
        """
        try:
            # Create a rect from bbox with some padding
            table_rect = fitz.Rect(bbox)

            # Add padding around the table (10 points on each side)
            padding = 10
            table_rect.x0 = max(0, table_rect.x0 - padding)
            table_rect.y0 = max(0, table_rect.y0 - padding)
            table_rect.x1 = min(page.rect.width, table_rect.x1 + padding)
            table_rect.y1 = min(page.rect.height, table_rect.y1 + padding)

            # Create pixmap of the table region
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat, clip=table_rect)

            # Handle colorspace issues similar to figure extraction
            if pix.colorspace is None:
                self.logger.warning(f"Skipping table with None colorspace: {table_id}")
                pix = None
                return None, None

            # Convert to RGB colorspace for consistent processing
            if pix.n - pix.alpha > 4:  # CMYK or other multi-channel
                pix = fitz.Pixmap(fitz.csRGB, pix)
            elif pix.n - pix.alpha == 1:  # Grayscale
                pix = fitz.Pixmap(fitz.csRGB, pix)
            elif pix.alpha:  # Has alpha channel
                pix = fitz.Pixmap(fitz.csRGB, pix)

            # Verify we have a valid pixmap
            if pix.colorspace is None:
                self.logger.warning(f"Skipping table with problematic colorspace: {table_id}")
                pix = None
                return None, None

            # Convert to PIL Image
            try:
                img_data = pix.tobytes("png")
                pil_image = Image.open(io.BytesIO(img_data))
            except Exception as conv_error:
                self.logger.warning(f"Failed to convert table pixmap to PIL Image: {conv_error}")
                pix = None
                return None, None

            # Save table image
            image_path = os.path.join(self.tables_dir, f"{table_id}_image.png")
            pil_image.save(image_path)

            # Generate CLIP embedding for the table image
            image_embedding = None
            try:
                image_embedding = self.clip_client.generate_image_embedding(pil_image)
            except Exception as embed_error:
                self.logger.warning(f"Failed to generate embedding for table image {table_id}: {embed_error}")

            # Clean up pixmap
            if pix:
                pix = None

            self.logger.info(f"Captured table image for {table_id}: {image_path}")
            return image_path, image_embedding

        except Exception as e:
            self.logger.error(f"Error capturing table image for {table_id}: {e}")
            # Ensure pixmap cleanup even in error cases
            try:
                if 'pix' in locals() and pix:
                    pix = None
            except:
                pass
            return None, None
