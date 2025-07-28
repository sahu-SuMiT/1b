"""
Relevance ranking module for prioritizing sections based on persona and job requirements.
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class RelevanceRanker:
    """Ranks sections by relevance to persona and job requirements."""
    
    def __init__(self):
        """Initialize the relevance ranker."""
        self.ranking_weights = {
            "relevance_score": 0.4,
            "section_importance": 0.2,
            "content_quality": 0.2,
            "document_diversity": 0.1,
            "coverage_balance": 0.1
        }
    
    def rank_sections(self, extracted_sections: List[Dict[str, Any]], persona_requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rank sections by relevance to persona and job requirements.
        
        Args:
            extracted_sections: List of extracted sections
            persona_requirements: Persona analysis results
            
        Returns:
            List of ranked sections with importance scores
        """
        logger.info(f"Ranking {len(extracted_sections)} extracted sections")
        
        if not extracted_sections:
            return []
        
        # Calculate ranking scores
        ranked_sections = []
        for section in extracted_sections:
            ranking_score = self._calculate_ranking_score(section, persona_requirements)
            section["importance_rank"] = ranking_score
            ranked_sections.append(section)
        
        # Sort by ranking score
        ranked_sections.sort(key=lambda x: x["importance_rank"], reverse=True)
        
        # Apply diversity and balance adjustments
        final_ranked = self._apply_diversity_balance(ranked_sections, persona_requirements)
        
        # Limit to top sections (avoid overwhelming output)
        max_sections = min(20, len(final_ranked))
        final_ranked = final_ranked[:max_sections]
        
        # Normalize importance ranks
        if final_ranked:
            max_rank = max(section["importance_rank"] for section in final_ranked)
            for section in final_ranked:
                section["importance_rank"] = round(section["importance_rank"] / max_rank, 3)
        
        logger.info(f"Ranked {len(final_ranked)} sections by relevance")
        return final_ranked
    
    def _calculate_ranking_score(self, section: Dict[str, Any], persona_requirements: Dict[str, Any]) -> float:
        """Calculate overall ranking score for a section."""
        total_score = 0.0
        total_weight = 0.0
        
        # Relevance score (from extraction)
        relevance_weight = self.ranking_weights["relevance_score"]
        relevance_score = section.get("relevance_score", 0.0)
        total_score += relevance_score * relevance_weight
        total_weight += relevance_weight
        
        # Section importance
        importance_weight = self.ranking_weights["section_importance"]
        importance_score = self._calculate_section_importance(section, persona_requirements)
        total_score += importance_score * importance_weight
        total_weight += importance_weight
        
        # Content quality
        quality_weight = self.ranking_weights["content_quality"]
        quality_score = self._calculate_content_quality(section)
        total_score += quality_score * quality_weight
        total_weight += quality_weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_section_importance(self, section: Dict[str, Any], persona_requirements: Dict[str, Any]) -> float:
        """Calculate section importance based on type and level."""
        section_type = section.get("section_type", "general")
        level = section.get("level", 1)
        
        # Get section priorities from persona requirements
        section_priorities = persona_requirements.get("section_priorities", {})
        type_priority = section_priorities.get(section_type, 0.5)
        
        # Level-based importance (higher levels are more important)
        level_importance = max(0.3, (4 - level) * 0.2)  # Level 1 = 0.6, Level 2 = 0.4, etc.
        
        # Combine type and level importance
        return (type_priority + level_importance) / 2.0
    
    def _calculate_content_quality(self, section: Dict[str, Any]) -> float:
        """Calculate content quality score."""
        content = section.get("content", "")
        title = section.get("title", "")
        
        # Length quality (optimal range)
        content_length = len(content)
        if content_length < 100:
            length_score = 0.3
        elif content_length < 500:
            length_score = 0.8
        elif content_length < 1500:
            length_score = 1.0
        else:
            length_score = 0.6  # Penalty for very long sections
        
        # Title quality
        title_length = len(title)
        if title_length < 10:
            title_score = 0.3
        elif title_length < 50:
            title_score = 0.8
        else:
            title_score = 0.5  # Penalty for very long titles
        
        # Key phrases quality
        key_phrases = section.get("key_phrases", [])
        phrase_score = min(1.0, len(key_phrases) / 5.0)  # More key phrases = better
        
        # Subsection quality
        subsections = section.get("subsections", [])
        subsection_score = min(1.0, len(subsections) / 3.0)  # Some subsections = good
        
        return (length_score + title_score + phrase_score + subsection_score) / 4.0
    
    def _apply_diversity_balance(self, ranked_sections: List[Dict[str, Any]], persona_requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply diversity and balance adjustments to ranking."""
        if len(ranked_sections) <= 5:
            return ranked_sections  # No need for diversity with small sets
        
        # Group sections by document
        document_groups = {}
        for section in ranked_sections:
            doc = section.get("document", "unknown")
            if doc not in document_groups:
                document_groups[doc] = []
            document_groups[doc].append(section)
        
        # Group sections by type
        type_groups = {}
        for section in ranked_sections:
            section_type = section.get("section_type", "general")
            if section_type not in type_groups:
                type_groups[section_type] = []
            type_groups[section_type].append(section)
        
        # Apply diversity adjustments
        diversity_weight = self.ranking_weights["document_diversity"]
        balance_weight = self.ranking_weights["coverage_balance"]
        
        for section in ranked_sections:
            # Document diversity bonus
            doc = section.get("document", "unknown")
            doc_count = len(document_groups.get(doc, []))
            if doc_count > 3:  # Penalty for over-representation
                diversity_penalty = min(0.2, (doc_count - 3) * 0.05)
                section["importance_rank"] *= (1 - diversity_penalty * diversity_weight)
            
            # Type balance bonus
            section_type = section.get("section_type", "general")
            type_count = len(type_groups.get(section_type, []))
            if type_count > 5:  # Penalty for over-representation
                balance_penalty = min(0.15, (type_count - 5) * 0.03)
                section["importance_rank"] *= (1 - balance_penalty * balance_weight)
        
        # Re-sort after adjustments
        ranked_sections.sort(key=lambda x: x["importance_rank"], reverse=True)
        
        return ranked_sections
    
    def _calculate_coverage_score(self, sections: List[Dict[str, Any]], persona_requirements: Dict[str, Any]) -> float:
        """Calculate how well the sections cover the required topics."""
        if not sections:
            return 0.0
        
        # Get required keywords
        relevant_keywords = persona_requirements.get("relevant_keywords", {})
        required_keywords = set(relevant_keywords.keys())
        
        if not required_keywords:
            return 0.5  # Neutral score if no specific keywords
        
        # Count covered keywords
        covered_keywords = set()
        for section in sections:
            content = section.get("content", "").lower()
            title = section.get("title", "").lower()
            
            for keyword in required_keywords:
                if keyword.lower() in content or keyword.lower() in title:
                    covered_keywords.add(keyword)
        
        return len(covered_keywords) / len(required_keywords)
    
    def _optimize_section_selection(self, ranked_sections: List[Dict[str, Any]], persona_requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optimize section selection for maximum coverage and relevance."""
        if len(ranked_sections) <= 10:
            return ranked_sections  # No optimization needed for small sets
        
        # Try different combinations to maximize coverage
        best_coverage = 0.0
        best_selection = ranked_sections[:10]  # Default to top 10
        
        # Simple greedy optimization
        for max_sections in [8, 10, 12, 15]:
            selection = ranked_sections[:max_sections]
            coverage = self._calculate_coverage_score(selection, persona_requirements)
            
            if coverage > best_coverage:
                best_coverage = coverage
                best_selection = selection
        
        logger.info(f"Optimized selection: {len(best_selection)} sections with {best_coverage:.2f} coverage")
        return best_selection 