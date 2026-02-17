"""
PDF Processing Handler for extracting figures and tables from academic papers.

This handler downloads PDFs, extracts visual content (figures and tables),
and generates descriptions and embeddings using various AI models.
"""

import logging
import os
import requests
from typing import List, Dict, Optional, Tuple
import fitz  # PyMuPDF
from PIL import Image
import io
import re
from datetime import datetime

from models.schemas.nodes import Figure, Table, Paper
from clients.huggingface.CLIPClient import CLIPClient
from clients.huggingface.DeepseekClient import DeepseekClient

class PDFProcessingHandler:
    """Handler for processing PDFs and extracting visual content."""
    
    def __init__(self, clip_client: Optional[CLIPClient] = None):
        self.clip_client = clip_client or CLIPClient()
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
                                
                                # Create Figure entity
                                figure = Figure(
                                    id=figure_id,
                                    paper_id=paper_id,
                                    figure_number=figure_counter,
                                    description=description,
                                    page_number=page_num + 1,
                                    image_path=image_path,
                                    image_embedding=image_embedding
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
                    table_text=table_text
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
        
        # Extract figures and tables
        figures, tables = self.extract_figures_and_tables(pdf_path, paper.id)
        
        # Clean up PDF file (optional)
        try:
            os.remove(pdf_path)
        except:
            pass
        
        return figures, tables