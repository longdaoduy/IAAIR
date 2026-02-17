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
    
    def __init__(self, clip_client: Optional[CLIPClient] = None, scibert_client: Optional[SciBERTClient] = None, milvus_client: Optional[MilvusClient] = None):
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
        Extract figures and tables from PDF.
        
        Args:
            pdf_path: Path to the PDF file
            paper_id: Paper ID for generating entity IDs
            
        Returns:
            Tuple of (figures, tables) lists
        """
        figures = []
        tables = []
        
        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            
            figure_counter = 1
            table_counter = 1
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Extract images (potential figures)
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Extract image
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            # Convert to PIL Image
                            if pix.alpha:
                                pix = fitz.Pixmap(fitz.csRGB, pix)
                            
                            img_data = pix.tobytes("png")
                            pil_image = Image.open(io.BytesIO(img_data))
                            
                            # Filter out small images (likely icons or decorations)
                            if pil_image.width >= 100 and pil_image.height >= 100:
                                # Save image
                                figure_id = f"{paper_id}#figure_{figure_counter}"
                                image_path = os.path.join(self.figures_dir, f"{figure_id}.png")
                                pil_image.save(image_path)
                                
                                # Generate CLIP embedding
                                image_embedding = self.clip_client.generate_image_embedding(pil_image)
                                
                                # Extract surrounding text for description
                                description = self._extract_figure_description(page, img, page_num + 1)
                                
                                # Generate SciBERT embedding for description
                                description_embedding = None
                                if description:
                                    description_embedding = self.scibert_client.generate_embedding(description)
                                
                                # Create Figure entity
                                figure = Figure(
                                    id=figure_id,
                                    paper_id=paper_id,
                                    figure_number=figure_counter,
                                    description=description,
                                    page_number=page_num + 1,
                                    image_path=image_path,
                                    image_embedding=image_embedding,
                                    description_embedding=description_embedding
                                )
                                
                                figures.append(figure)
                                figure_counter += 1
                        
                        pix = None  # Clean up
                        
                    except Exception as e:
                        self.logger.error(f"Error processing image on page {page_num + 1}: {e}")
                        continue
                
                # Extract tables
                page_tables = self._extract_tables_from_page(page, paper_id, table_counter, page_num + 1)
                tables.extend(page_tables)
                table_counter += len(page_tables)
            
            doc.close()
            
            self.logger.info(f"Extracted {len(figures)} figures and {len(tables)} tables from {paper_id}")
            
        except Exception as e:
            self.logger.error(f"Error processing PDF {pdf_path}: {e}")
        
        return figures, tables
    
    def _extract_figure_description(self, page, img_info, page_num: int) -> Optional[str]:
        """
        Extract caption/description for a figure by analyzing surrounding text.
        
        Args:
            page: PyMuPDF page object
            img_info: Image information from PyMuPDF
            page_num: Page number
            
        Returns:
            Figure description if found
        """
        try:
            # Get image position
            img_rect = page.get_image_rects(img_info[0])[0] if page.get_image_rects(img_info[0]) else None
            
            if not img_rect:
                return None
            
            # Extract text from the page
            text_instances = page.get_text("dict")
            
            # Look for figure captions near the image
            potential_captions = []
            
            for block in text_instances["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        line_rect = fitz.Rect(line["bbox"])
                        
                        for span in line["spans"]:
                            line_text += span["text"]
                        
                        line_text = line_text.strip()
                        
                        # Check if text contains figure references
                        if re.search(r'(fig|figure)\s*\d+', line_text.lower()):
                            # Calculate distance from image
                            distance = self._calculate_distance(img_rect, line_rect)
                            potential_captions.append((line_text, distance))
            
            # Return the closest caption
            if potential_captions:
                potential_captions.sort(key=lambda x: x[1])
                return potential_captions[0][0]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting figure description: {e}")
            return None
    
    def _extract_tables_from_page(self, page, paper_id: str, start_counter: int, page_num: int) -> List[Table]:
        """
        Extract tables from a page using text analysis.
        
        Args:
            page: PyMuPDF page object
            paper_id: Paper ID
            start_counter: Starting table counter
            page_num: Page number
            
        Returns:
            List of Table entities
        """
        tables = []
        
        try:
            # Get page text with formatting
            text_dict = page.get_text("dict")
            
            # Look for table-like structures (this is a simplified approach)
            # In practice, you might want to use more sophisticated table detection
            table_patterns = []
            
            for block in text_dict["blocks"]:
                if "lines" in block:
                    block_text = ""
                    for line in block["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            line_text += span["text"]
                        block_text += line_text + "\n"
                    
                    # Simple heuristic: look for patterns that might be tables
                    if self._is_likely_table(block_text):
                        table_patterns.append((block_text.strip(), block["bbox"]))
            
            # Create table entities
            for i, (table_text, bbox) in enumerate(table_patterns):
                table_id = f"{paper_id}#table_{start_counter + i}"
                
                # Extract table description/caption
                description = self._extract_table_description(page, bbox)
                
                # Generate SciBERT embedding for description
                description_embedding = None
                if description:
                    description_embedding = self.scibert_client.generate_embedding(description)
                
                # Try to parse table structure
                headers, rows = self._parse_table_structure(table_text)
                
                table = Table(
                    id=table_id,
                    paper_id=paper_id,
                    table_number=start_counter + i,
                    description=description,
                    page_number=page_num,
                    headers=headers,
                    rows=rows,
                    table_text=table_text,
                    description_embedding=description_embedding
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
                if figure.description_embedding or figure.image_embedding:
                    # Generate sparse embedding for description if available
                    sparse_description_embedding = {0: 0.001}  # Default minimal sparse vector
                    if figure.description and self.milvus_client.is_tfidf_fitted:
                        sparse_description_embedding = self.milvus_client.generate_sparse_embedding(figure.description)
                    
                    figure_data = {
                        'id': figure.id,
                        'paper_id': figure.paper_id,
                        'description': figure.description or "",
                        'description_embedding': figure.description_embedding,
                        'image_embedding': figure.image_embedding,
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
                self.logger.info(f"Successfully saved {len(figures_data)} figures to dedicated Milvus figures collection")
            else:
                self.logger.error("Failed to save figures to Milvus")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error saving figures to Milvus: {e}")
            return False

    def _save_tables_to_milvus(self, tables: List[Table]) -> bool:
        """
        Save table embeddings to Milvus using existing MilvusClient functionality.
        
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
            
            # Prepare tables as embedding data for MilvusClient
            embedding_data = []
            for table in tables:
                if table.description_embedding:
                    embedding_item = {
                        'paper_id': table.id,  # Use table ID as paper_id for unique identification
                        'embedding': table.description_embedding
                    }
                    embedding_data.append(embedding_item)
            
            if not embedding_data:
                self.logger.warning("No table embeddings to save to Milvus")
                return True
            
            # Prepare papers_data for text extraction (for sparse embeddings)
            papers_data = []
            for table in tables:
                if table.description_embedding:
                    paper_data = {
                        'paper': {
                            'id': table.id,
                            'title': f"Table {table.table_number}",
                            'abstract': table.description or table.table_text or f"Table {table.table_number} from paper {table.paper_id}"
                        }
                    }
                    papers_data.append(paper_data)
            
            # Create temporary embedding file
            temp_filename = f"temp_tables_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(temp_filename, 'w', encoding='utf-8') as f:
                json.dump(embedding_data, f, indent=2)
            
            # Upload using existing MilvusClient method
            success = self.milvus_client.upload_embeddings(
                embedding_file=temp_filename,
                papers_data=papers_data,
                batch_size=100
            )
            
            # Clean up temp file
            try:
                os.remove(temp_filename)
            except:
                pass
            
            if success:
                self.logger.info(f"Successfully saved {len(embedding_data)} tables to Milvus")
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
            
            # Save tables to existing papers collection
            tables_success = self._save_tables_to_milvus(tables)
            if not tables_success:
                self.logger.warning("Failed to save some tables to Milvus")
                
            if figures_success and tables_success:
                self.logger.info("Successfully saved all figures to dedicated collection and tables to papers collection")
            
        except Exception as e:
            self.logger.error(f"Error saving to Milvus: {e}")
        
        # Clean up PDF file (optional)
        try:
            os.remove(pdf_path)
        except:
            pass
        
        return figures, tables