"""
Document processing module for extracting text and structure from PDF documents.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Processes PDF documents to extract text and structural information."""
    
    def __init__(self):
        """Initialize the document processor."""
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        self.stop_words = set(stopwords.words('english'))
        
        # Section detection patterns
        self.section_patterns = [
            r'^\d+\.\s+[A-Z][^.]*',  # 1. Section Title
            r'^\d+\.\d+\s+[A-Z][^.]*',  # 1.1 Subsection Title
            r'^[A-Z][A-Z\s]{2,}$',  # ALL CAPS TITLES
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$',  # Title Case
        ]
        
    def process_document(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Process a PDF document and extract structured content.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing document structure and content
        """
        logger.info(f"Processing document: {pdf_path.name}")
        
        document_data = {
            "title": "",
            "pages": [],
            "sections": [],
            "metadata": {}
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Extract metadata
                document_data["metadata"] = {
                    "num_pages": len(pdf.pages),
                    "filename": pdf_path.name
                }
                
                # Process each page
                for page_num, page in enumerate(pdf.pages, 1):
                    page_data = self._process_page(page, page_num)
                    document_data["pages"].append(page_data)
                
                # Extract title from first page
                if document_data["pages"]:
                    document_data["title"] = self._extract_title(document_data["pages"][0])
                
                # Identify sections across pages
                document_data["sections"] = self._identify_sections(document_data["pages"])
                
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            raise
        
        logger.info(f"Extracted {len(document_data['sections'])} sections from {pdf_path.name}")
        return document_data
    
    def _process_page(self, page, page_num: int) -> Dict[str, Any]:
        """Process a single page and extract text and formatting information."""
        page_data = {
            "page_number": page_num,
            "text": "",
            "lines": [],
            "font_info": [],
            "sections": []
        }
        
        # Extract text
        text = page.extract_text()
        page_data["text"] = text if text else ""
        
        # Extract lines with font information
        if page.chars:
            lines = self._extract_lines_with_fonts(page)
            page_data["lines"] = lines
            
            # Identify sections on this page
            page_data["sections"] = self._identify_page_sections(lines)
        
        return page_data
    
    def _extract_lines_with_fonts(self, page) -> List[Dict[str, Any]]:
        """Extract lines with font information from page."""
        lines = []
        
        # Group characters by line
        char_groups = {}
        for char in page.chars:
            if char['text'].strip():
                y_pos = round(char['top'], 2)
                if y_pos not in char_groups:
                    char_groups[y_pos] = []
                char_groups[y_pos].append(char)
        
        # Sort by y position (top to bottom)
        for y_pos in sorted(char_groups.keys()):
            chars = char_groups[y_pos]
            chars.sort(key=lambda x: x['x0'])  # Sort by x position
            
            # Combine characters into line
            line_text = ''.join(char['text'] for char in chars)
            if line_text.strip():
                # Get font information
                fonts = [char.get('fontname', '') for char in chars if char.get('fontname')]
                font_sizes = [char.get('size', 0) for char in chars if char.get('size')]
                
                line_data = {
                    "text": line_text.strip(),
                    "y_position": y_pos,
                    "x_position": chars[0]['x0'] if chars else 0,
                    "font_name": fonts[0] if fonts else "",
                    "font_size": max(font_sizes) if font_sizes else 0,
                    "is_bold": any('bold' in font.lower() for font in fonts),
                    "is_likely_heading": self._is_likely_heading(line_text, fonts, font_sizes)
                }
                lines.append(line_data)
        
        return lines
    
    def _is_likely_heading(self, text: str, fonts: List[str], font_sizes: List[float]) -> bool:
        """Determine if a line is likely a heading based on text and formatting."""
        if not text.strip():
            return False
        
        # Check for heading patterns
        for pattern in self.section_patterns:
            if re.match(pattern, text.strip()):
                return True
        
        # Check for formatting indicators
        if any('bold' in font.lower() for font in fonts):
            return True
        
        # Check for larger font size
        if font_sizes and max(font_sizes) > 12:  # Assuming normal text is 10-12pt
            return True
        
        # Check for short, capitalized text
        words = text.strip().split()
        if len(words) <= 8 and text.strip().isupper():
            return True
        
        return False
    
    def _extract_title(self, first_page: Dict[str, Any]) -> str:
        """Extract document title from the first page."""
        if not first_page.get("lines"):
            return ""
        
        # Look for the first prominent heading
        for line in first_page["lines"]:
            if line.get("is_likely_heading") and line.get("y_position", 0) < 200:
                return line["text"]
        
        # Fallback: first line with significant text
        for line in first_page["lines"]:
            if len(line["text"]) > 10 and line.get("y_position", 0) < 300:
                return line["text"]
        
        return ""
    
    def _identify_page_sections(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify sections on a single page."""
        sections = []
        current_section = None
        
        for line in lines:
            if line.get("is_likely_heading"):
                # Save previous section if exists
                if current_section:
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    "title": line["text"],
                    "start_line": line,
                    "content_lines": [],
                    "level": self._determine_section_level(line["text"])
                }
            elif current_section:
                current_section["content_lines"].append(line)
        
        # Add final section
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _determine_section_level(self, title: str) -> int:
        """Determine the hierarchical level of a section."""
        # Check for numbered patterns
        if re.match(r'^\d+\.\d+\.\d+', title):
            return 3
        elif re.match(r'^\d+\.\d+', title):
            return 2
        elif re.match(r'^\d+\.', title):
            return 1
        else:
            return 1  # Default to level 1
    
    def _identify_sections(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify sections across all pages."""
        all_sections = []
        
        for page in pages:
            for section in page.get("sections", []):
                section_data = {
                    "title": section["title"],
                    "level": section["level"],
                    "page_number": page["page_number"],
                    "content": self._extract_section_content(section["content_lines"]),
                    "start_position": section["start_line"]["y_position"],
                    "end_position": self._get_section_end_position(section["content_lines"])
                }
                all_sections.append(section_data)
        
        return all_sections
    
    def _extract_section_content(self, content_lines: List[Dict[str, Any]]) -> str:
        """Extract text content from section lines."""
        return " ".join(line["text"] for line in content_lines if line["text"].strip())
    
    def _get_section_end_position(self, content_lines: List[Dict[str, Any]]) -> float:
        """Get the end position of a section."""
        if not content_lines:
            return 0
        return content_lines[-1]["y_position"] 