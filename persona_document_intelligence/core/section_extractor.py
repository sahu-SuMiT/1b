"""
Section extraction module for identifying and extracting relevant sections from documents.
"""

import logging
import re
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class SectionExtractor:
    """Extracts relevant sections from documents based on persona requirements."""
    
    def __init__(self):
        """Initialize the section extractor."""
        # Initialize sentence transformer for semantic similarity
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence transformer model")
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
            self.sentence_model = None
        
        # Section identification patterns
        self.section_patterns = {
            "methodology": [
                r"methodology", r"methods?", r"approach", r"procedure", r"technique",
                r"experimental", r"design", r"implementation", r"algorithm"
            ],
            "results": [
                r"results?", r"findings", r"outcomes?", r"performance", r"evaluation",
                r"analysis", r"data", r"statistics", r"measurements?"
            ],
            "conclusion": [
                r"conclusion", r"summary", r"discussion", r"implications?", r"recommendations?",
                r"future work", r"limitations?", r"outlook", r"final"
            ],
            "introduction": [
                r"introduction", r"background", r"overview", r"context", r"problem",
                r"motivation", r"objectives?", r"goals?", r"scope"
            ],
            "literature": [
                r"literature", r"related work", r"previous", r"existing", r"state of the art",
                r"survey", r"review", r"background", r"prior work"
            ]
        }
        
    def extract_sections(self, document_content: Dict[str, Any], persona_requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract relevant sections from document content based on persona requirements.
        
        Args:
            document_content: Processed document content
            persona_requirements: Persona analysis results
            
        Returns:
            List of extracted sections with relevance information
        """
        logger.info(f"Extracting sections from document: {document_content.get('title', 'Unknown')}")
        
        extracted_sections = []
        
        # Get relevance criteria
        relevance_criteria = persona_requirements.get("relevance_criteria", {})
        content_filters = persona_requirements.get("content_filters", {})
        
        # Process each section in the document
        for section in document_content.get("sections", []):
            # Apply content filters
            if not self._passes_content_filters(section, content_filters):
                continue
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(section, relevance_criteria)
            
            # Only include sections with sufficient relevance
            if relevance_score > 0.3:  # Minimum relevance threshold
                section_data = {
                    "title": section.get("title", ""),
                    "content": section.get("content", ""),
                    "page_number": section.get("page_number", 0),
                    "level": section.get("level", 1),
                    "relevance_score": relevance_score,
                    "section_type": self._identify_section_type(section.get("title", "")),
                    "key_phrases": self._extract_key_phrases(section.get("content", "")),
                    "start_position": section.get("start_position", 0),
                    "end_position": section.get("end_position", 0)
                }
                
                # Extract subsections
                subsections = self._extract_subsections(section, relevance_criteria)
                section_data["subsections"] = subsections
                
                extracted_sections.append(section_data)
        
        # Sort by relevance score
        extracted_sections.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        logger.info(f"Extracted {len(extracted_sections)} relevant sections")
        return extracted_sections
    
    def _passes_content_filters(self, section: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if section passes content filters."""
        content = section.get("content", "")
        title = section.get("title", "")
        
        # Check length constraints
        min_length = filters.get("min_section_length", 50)
        max_length = filters.get("max_section_length", 2000)
        
        if len(content) < min_length or len(content) > max_length:
            return False
        
        # Check exclusion patterns
        exclude_patterns = filters.get("exclude_patterns", [])
        for pattern in exclude_patterns:
            if re.search(pattern, title.lower()):
                return False
        
        return True
    
    def _calculate_relevance_score(self, section: Dict[str, Any], criteria: Dict[str, Any]) -> float:
        """Calculate relevance score for a section."""
        total_score = 0.0
        total_weight = 0.0
        
        # Keyword matching
        keyword_criteria = criteria.get("keyword_matching", {})
        keyword_weight = keyword_criteria.get("weight", 0.4)
        keywords = keyword_criteria.get("keywords", {})
        
        keyword_score = self._calculate_keyword_score(section, keywords)
        total_score += keyword_score * keyword_weight
        total_weight += keyword_weight
        
        # Section relevance
        section_criteria = criteria.get("section_relevance", {})
        section_weight = section_criteria.get("weight", 0.3)
        priorities = section_criteria.get("priorities", {})
        
        section_score = self._calculate_section_type_score(section, priorities)
        total_score += section_score * section_weight
        total_weight += section_weight
        
        # Content quality
        quality_criteria = criteria.get("content_quality", {})
        quality_weight = quality_criteria.get("weight", 0.2)
        
        quality_score = self._calculate_content_quality_score(section, quality_criteria)
        total_score += quality_score * quality_weight
        total_weight += quality_weight
        
        # Job specificity
        job_criteria = criteria.get("job_specificity", {})
        job_weight = job_criteria.get("weight", 0.1)
        specific_keywords = job_criteria.get("specific_keywords", set())
        
        job_score = self._calculate_job_specificity_score(section, specific_keywords)
        total_score += job_score * job_weight
        total_weight += job_weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_keyword_score(self, section: Dict[str, Any], keywords: Dict[str, float]) -> float:
        """Calculate keyword matching score."""
        content = section.get("content", "").lower()
        title = section.get("title", "").lower()
        
        total_score = 0.0
        total_weight = 0.0
        
        for keyword, weight in keywords.items():
            # Check in title (higher weight)
            title_matches = len(re.findall(r'\b' + re.escape(keyword) + r'\b', title))
            if title_matches > 0:
                total_score += weight * 2.0  # Double weight for title matches
                total_weight += weight
            
            # Check in content
            content_matches = len(re.findall(r'\b' + re.escape(keyword) + r'\b', content))
            if content_matches > 0:
                total_score += weight * min(content_matches / 10.0, 1.0)  # Normalize by frequency
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_section_type_score(self, section: Dict[str, Any], priorities: Dict[str, float]) -> float:
        """Calculate section type relevance score."""
        section_type = self._identify_section_type(section.get("title", ""))
        
        # Get priority for this section type
        priority = priorities.get(section_type, 0.5)
        
        # Additional boost for higher-level sections
        level = section.get("level", 1)
        level_boost = max(0.1, (4 - level) * 0.1)  # Higher levels get more boost
        
        return min(1.0, priority + level_boost)
    
    def _calculate_content_quality_score(self, section: Dict[str, Any], criteria: Dict[str, Any]) -> float:
        """Calculate content quality score."""
        content = section.get("content", "")
        
        # Length-based scoring
        min_length = criteria.get("min_length", 50)
        max_length = criteria.get("max_length", 2000)
        
        if len(content) < min_length:
            return 0.0
        elif len(content) > max_length:
            return 0.5  # Penalty for very long sections
        else:
            # Optimal length range
            return 1.0
    
    def _calculate_job_specificity_score(self, section: Dict[str, Any], specific_keywords: set) -> float:
        """Calculate job-specific keyword score."""
        if not specific_keywords:
            return 0.5  # Neutral score if no specific keywords
        
        content = section.get("content", "").lower()
        title = section.get("title", "").lower()
        
        matches = 0
        for keyword in specific_keywords:
            if keyword.lower() in title or keyword.lower() in content:
                matches += 1
        
        return min(1.0, matches / len(specific_keywords))
    
    def _identify_section_type(self, title: str) -> str:
        """Identify the type of section based on title."""
        title_lower = title.lower()
        
        for section_type, patterns in self.section_patterns.items():
            for pattern in patterns:
                if re.search(pattern, title_lower):
                    return section_type
        
        return "general"
    
    def _extract_key_phrases(self, content: str) -> List[str]:
        """Extract key phrases from content."""
        # Simple key phrase extraction based on frequency and importance
        words = re.findall(r'\b\w+\b', content.lower())
        
        # Filter out common words
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can"}
        filtered_words = [word for word in words if word not in common_words and len(word) > 3]
        
        # Count frequency
        from collections import Counter
        word_counts = Counter(filtered_words)
        
        # Return top phrases
        return [word for word, count in word_counts.most_common(10)]
    
    def _extract_subsections(self, section: Dict[str, Any], relevance_criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract relevant subsections from a section."""
        content = section.get("content", "")
        
        # Simple subsection extraction based on sentences
        sentences = re.split(r'[.!?]+', content)
        subsections = []
        
        current_subsection = ""
        sentence_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            current_subsection += sentence + ". "
            sentence_count += 1
            
            # Create subsection every 3-5 sentences
            if sentence_count >= 3 and len(current_subsection) > 100:
                # Calculate relevance for this subsection
                subsection_score = self._calculate_relevance_score({
                    "title": f"Subsection of {section.get('title', '')}",
                    "content": current_subsection
                }, relevance_criteria)
                
                if subsection_score > 0.2:  # Lower threshold for subsections
                    subsections.append({
                        "content": current_subsection.strip(),
                        "relevance_score": subsection_score,
                        "key_phrases": self._extract_key_phrases(current_subsection)
                    })
                
                current_subsection = ""
                sentence_count = 0
        
        # Add remaining content if significant
        if current_subsection and len(current_subsection) > 50:
            subsection_score = self._calculate_relevance_score({
                "title": f"Subsection of {section.get('title', '')}",
                "content": current_subsection
            }, relevance_criteria)
            
            if subsection_score > 0.2:
                subsections.append({
                    "content": current_subsection.strip(),
                    "relevance_score": subsection_score,
                    "key_phrases": self._extract_key_phrases(current_subsection)
                })
        
        return subsections 