"""
Output formatting module for generating the required JSON output format.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class OutputFormatter:
    """Formats extracted sections into the required JSON output format."""
    
    def __init__(self):
        """Initialize the output formatter."""
        pass
    
    def format_output(self, documents: List[str], persona: str, job_to_be_done: str, 
                     ranked_sections: List[Dict[str, Any]], processing_timestamp: float,
                     subsection_analysis: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Format the extracted sections into the required JSON output format.
        
        Args:
            documents: List of input document paths
            persona: Persona description
            job_to_be_done: Job to be done description
            ranked_sections: List of ranked sections
            processing_timestamp: Processing timestamp
            
        Returns:
            Formatted JSON output
        """
        logger.info("Formatting output to required JSON structure")
        
        # Convert timestamp to readable format
        timestamp = datetime.fromtimestamp(processing_timestamp).isoformat()
        
        # Format extracted sections
        extracted_sections = []
        for i, section in enumerate(ranked_sections):
            section_data = {
                "document": section.get("document", ""),
                "section_title": section.get("section_title", section.get("title", "")),
                "importance_rank": i + 1,  # Use sequential ranking (1, 2, 3, ...)
                "page_number": section.get("page_number", 0)
            }
            extracted_sections.append(section_data)
        
        # Format subsection analysis
        if subsection_analysis:
            # Use the provided subsection analysis
            subsection_data = []
            for subsection in subsection_analysis:
                subsection_item = {
                    "document": subsection.get("document", ""),
                    "refined_text": subsection.get("refined_text", subsection.get("content", "")),
                    "page_number": subsection.get("page_number", 0)
                }
                subsection_data.append(subsection_item)
            subsection_analysis_output = subsection_data
        else:
            # Legacy fallback: extract from sections
            subsection_analysis_output = []
            for section in ranked_sections:
                document = section.get("document", "")
                subsections = section.get("subsections", [])
                
                for subsection in subsections:
                    subsection_data = {
                        "document": document,
                        "refined_text": subsection.get("content", ""),
                        "page_number": section.get("page_number", 0)
                    }
                    subsection_analysis_output.append(subsection_data)
        
        # Build the complete output structure
        output = {
            "metadata": {
                "input_documents": documents,
                "persona": persona,
                "job_to_be_done": job_to_be_done,
                "processing_timestamp": timestamp
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis_output
        }
        
        logger.info(f"Formatted output with {len(extracted_sections)} sections and {len(subsection_analysis_output)} subsections")
        return output
    
    def format_section_for_output(self, section: Dict[str, Any]) -> Dict[str, Any]:
        """Format a single section for output."""
        return {
            "document": section.get("document", ""),
            "section_title": section.get("title", ""),
            "importance_rank": section.get("importance_rank", 0.0),
            "page_number": section.get("page_number", 0),
            "section_type": section.get("section_type", "general"),
            "relevance_score": section.get("relevance_score", 0.0),
            "key_phrases": section.get("key_phrases", []),
            "content_length": len(section.get("content", ""))
        }
    
    def format_subsection_for_output(self, subsection: Dict[str, Any], parent_section: Dict[str, Any]) -> Dict[str, Any]:
        """Format a single subsection for output."""
        return {
            "document": parent_section.get("document", ""),
            "refined_text": subsection.get("content", ""),
            "page_number": parent_section.get("page_number", 0),
            "relevance_score": subsection.get("relevance_score", 0.0),
            "key_phrases": subsection.get("key_phrases", [])
        }
    
    def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate the output format."""
        required_fields = ["metadata", "extracted_sections", "subsection_analysis"]
        
        # Check required top-level fields
        for field in required_fields:
            if field not in output:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Check metadata fields
        metadata = output.get("metadata", {})
        metadata_fields = ["input_documents", "persona", "job_to_be_done", "processing_timestamp"]
        for field in metadata_fields:
            if field not in metadata:
                logger.error(f"Missing metadata field: {field}")
                return False
        
        # Check extracted sections format
        extracted_sections = output.get("extracted_sections", [])
        if not isinstance(extracted_sections, list):
            logger.error("extracted_sections must be a list")
            return False
        
        for section in extracted_sections:
            if not self._validate_section_format(section):
                return False
        
        # Check subsection analysis format
        subsection_analysis = output.get("subsection_analysis", [])
        if not isinstance(subsection_analysis, list):
            logger.error("subsection_analysis must be a list")
            return False
        
        for subsection in subsection_analysis:
            if not self._validate_subsection_format(subsection):
                return False
        
        logger.info("Output validation passed")
        return True
    
    def _validate_section_format(self, section: Dict[str, Any]) -> bool:
        """Validate section format."""
        required_fields = ["document", "section_title", "importance_rank", "page_number"]
        
        for field in required_fields:
            if field not in section:
                logger.error(f"Missing section field: {field}")
                return False
        
        # Validate data types
        if not isinstance(section.get("document"), str):
            logger.error("document must be a string")
            return False
        
        if not isinstance(section.get("section_title"), str):
            logger.error("section_title must be a string")
            return False
        
        if not isinstance(section.get("importance_rank"), (int, float)):
            logger.error("importance_rank must be a number")
            return False
        
        if not isinstance(section.get("page_number"), int):
            logger.error("page_number must be an integer")
            return False
        
        return True
    
    def _validate_subsection_format(self, subsection: Dict[str, Any]) -> bool:
        """Validate subsection format."""
        required_fields = ["document", "refined_text", "page_number"]
        
        for field in required_fields:
            if field not in subsection:
                logger.error(f"Missing subsection field: {field}")
                return False
        
        # Validate data types
        if not isinstance(subsection.get("document"), str):
            logger.error("document must be a string")
            return False
        
        if not isinstance(subsection.get("refined_text"), str):
            logger.error("refined_text must be a string")
            return False
        
        if not isinstance(subsection.get("page_number"), int):
            logger.error("page_number must be an integer")
            return False
        
        return True
    
    def generate_summary(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the output."""
        extracted_sections = output.get("extracted_sections", [])
        subsection_analysis = output.get("subsection_analysis", [])
        
        # Count sections by document
        document_counts = {}
        for section in extracted_sections:
            doc = section.get("document", "unknown")
            document_counts[doc] = document_counts.get(doc, 0) + 1
        
        # Calculate average importance rank
        importance_ranks = [section.get("importance_rank", 0) for section in extracted_sections]
        avg_importance = sum(importance_ranks) / len(importance_ranks) if importance_ranks else 0
        
        # Count subsections by document
        subsection_counts = {}
        for subsection in subsection_analysis:
            doc = subsection.get("document", "unknown")
            subsection_counts[doc] = subsection_counts.get(doc, 0) + 1
        
        summary = {
            "total_sections": len(extracted_sections),
            "total_subsections": len(subsection_analysis),
            "documents_processed": len(document_counts),
            "average_importance_rank": round(avg_importance, 3),
            "sections_per_document": document_counts,
            "subsections_per_document": subsection_counts,
            "processing_timestamp": output.get("metadata", {}).get("processing_timestamp", "")
        }
        
        return summary 