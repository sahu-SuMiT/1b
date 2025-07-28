"""
Advanced Subsection Analyzer for Challenge 1B
Provides granular text refinement and page-level analysis
"""

import re
import logging
from typing import Dict, List, Any, Tuple
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk

logger = logging.getLogger(__name__)

class AdvancedSubsectionAnalyzer:
    """Advanced analyzer for extracting and refining subsections."""
    
    def __init__(self):
        """Initialize the subsection analyzer."""
        # Ensure NLTK data is available
        for dataset in ['punkt', 'stopwords']:
            try:
                nltk.data.find(f'tokenizers/{dataset}' if dataset == 'punkt' else f'corpora/{dataset}')
            except LookupError:
                nltk.download(dataset, quiet=True)
        
        self.stop_words = set(stopwords.words('english'))
        
        # Refinement patterns based on persona types
        self.refinement_patterns = {
            "travel_planner": {
                "extract_patterns": [
                    r"(activities?|things to do|attractions?)[:\-]?\s*([^.]*(?:\.[^.]*){0,3})",
                    r"(restaurants?|dining|food)[:\-]?\s*([^.]*(?:\.[^.]*){0,3})",
                    r"(hotels?|accommodation|lodging)[:\-]?\s*([^.]*(?:\.[^.]*){0,3})",
                    r"(tips?|advice|recommendations?)[:\-]?\s*([^.]*(?:\.[^.]*){0,3})",
                    r"(transportation|getting around)[:\-]?\s*([^.]*(?:\.[^.]*){0,3})"
                ],
                "enhancement_keywords": ["visit", "enjoy", "experience", "explore", "discover", "try"],
                "list_indicators": ["•", "-", "*", "1.", "2.", "3.", "first", "second", "next"]
            },
            "hr_professional": {
                "extract_patterns": [
                    r"(steps?|process|procedure)[:\-]?\s*([^.]*(?:\.[^.]*){0,5})",
                    r"(forms?|fields?|fillable)[:\-]?\s*([^.]*(?:\.[^.]*){0,4})",
                    r"(signatures?|signing|approval)[:\-]?\s*([^.]*(?:\.[^.]*){0,3})",
                    r"(workflow|process|procedure)[:\-]?\s*([^.]*(?:\.[^.]*){0,4})",
                    r"(compliance|requirements?)[:\-]?\s*([^.]*(?:\.[^.]*){0,3})"
                ],
                "enhancement_keywords": ["create", "manage", "configure", "setup", "enable", "process"],
                "list_indicators": ["1)", "2)", "step", "first", "then", "next", "finally"]
            },
            "food_contractor": {
                "extract_patterns": [
                    r"(recipes?|ingredients?|cooking)[:\-]?\s*([^.]*(?:\.[^.]*){0,4})",
                    r"(vegetarian|vegan|plant-based)[:\-]?\s*([^.]*(?:\.[^.]*){0,3})",
                    r"(gluten[- ]free|dairy[- ]free)[:\-]?\s*([^.]*(?:\.[^.]*){0,3})",
                    r"(buffet|serving|portions?)[:\-]?\s*([^.]*(?:\.[^.]*){0,3})",
                    r"(preparation|prep|cooking)[:\-]?\s*([^.]*(?:\.[^.]*){0,4})"
                ],
                "enhancement_keywords": ["prepare", "cook", "serve", "combine", "mix", "add"],
                "list_indicators": ["•", "-", "ingredients:", "serves", "portion", "recipe"]
            }
        }
        
        # Content enhancement rules
        self.enhancement_rules = {
            "context_expansion": True,  # Add context around key information
            "list_formatting": True,    # Format lists and bullet points nicely
            "sentence_completion": True, # Complete partial sentences
            "redundancy_removal": True,  # Remove redundant information
            "detail_preservation": True  # Preserve important details
        }
    
    def analyze_subsections(self, sections: List[Dict[str, Any]], 
                          persona_type: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze and refine subsections from selected sections.
        
        Args:
            sections: List of selected high-relevance sections
            persona_type: Type of persona for context-aware refinement
            documents: Original document data for context
            
        Returns:
            List of refined subsection analyses
        """
        logger.info(f"Starting subsection analysis for {len(sections)} sections")
        
        subsection_analyses = []
        
        for section in sections:
            # Extract and refine subsections for this section
            refined_subsections = self._extract_refined_subsections(
                section, persona_type, documents
            )
            
            subsection_analyses.extend(refined_subsections)
        
        # Sort by relevance and select best subsections
        sorted_subsections = sorted(
            subsection_analyses, 
            key=lambda x: x.get("refinement_score", 0.0), 
            reverse=True
        )
        
        # Select top subsections (up to 5 as per challenge requirements)
        final_subsections = sorted_subsections[:5]
        
        logger.info(f"Generated {len(final_subsections)} refined subsections")
        return final_subsections
    
    def _extract_refined_subsections(self, section: Dict[str, Any], 
                                   persona_type: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract and refine subsections from a single section."""
        content = section["content"]
        document_name = section["document"]
        page_number = section["page_number"]
        
        # Get document context for enhanced refinement
        doc_context = self._get_document_context(document_name, documents)
        
        # Extract potential subsections using multiple methods
        potential_subsections = []
        
        # Method 1: Pattern-based extraction
        pattern_subsections = self._extract_pattern_based_subsections(
            content, persona_type, document_name, page_number
        )
        potential_subsections.extend(pattern_subsections)
        
        # Method 2: Sentence-level analysis
        sentence_subsections = self._extract_sentence_based_subsections(
            content, persona_type, document_name, page_number, doc_context
        )
        potential_subsections.extend(sentence_subsections)
        
        # Method 3: List and structured content extraction
        structured_subsections = self._extract_structured_subsections(
            content, persona_type, document_name, page_number
        )
        potential_subsections.extend(structured_subsections)
        
        # Refine and enhance the extracted subsections
        refined_subsections = []
        for subsection in potential_subsections:
            refined = self._refine_subsection_content(subsection, persona_type, doc_context)
            if refined and self._validate_subsection_quality(refined):
                refined_subsections.append(refined)
        
        # Remove duplicates and merge similar content
        deduplicated_subsections = self._deduplicate_subsections(refined_subsections)
        
        return deduplicated_subsections
    
    def _get_document_context(self, document_name: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get additional context from the source document."""
        for doc in documents:
            if doc.get("metadata", {}).get("filename") == document_name:
                return {
                    "title": doc.get("title", ""),
                    "full_content": self._extract_full_document_text(doc),
                    "sections": doc.get("sections", []),
                    "metadata": doc.get("metadata", {})
                }
        return {}
    
    def _extract_full_document_text(self, doc: Dict[str, Any]) -> str:
        """Extract full text content from document."""
        full_text = ""
        for page in doc.get("pages", []):
            full_text += page.get("text", "") + " "
        return full_text.strip()
    
    def _extract_pattern_based_subsections(self, content: str, persona_type: str, 
                                         document_name: str, page_number: int) -> List[Dict[str, Any]]:
        """Extract subsections using persona-specific patterns."""
        subsections = []
        
        if persona_type in self.refinement_patterns:
            patterns = self.refinement_patterns[persona_type]["extract_patterns"]
            
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    if len(match.groups()) >= 2:
                        topic = match.group(1).strip()
                        extracted_content = match.group(2).strip()
                        
                        if len(extracted_content) > 30:  # Minimum content length
                            subsection = {
                                "document": document_name,
                                "page_number": page_number,
                                "topic": topic,
                                "raw_content": extracted_content,
                                "extraction_method": "pattern_based",
                                "refinement_score": 0.7  # Base score for pattern matches
                            }
                            subsections.append(subsection)
        
        return subsections
    
    def _extract_sentence_based_subsections(self, content: str, persona_type: str, 
                                          document_name: str, page_number: int, 
                                          doc_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract subsections by analyzing sentence clusters."""
        sentences = sent_tokenize(content)
        subsections = []
        
        if len(sentences) < 3:
            return subsections
        
        # Group sentences into meaningful clusters
        current_cluster = []
        cluster_topic = ""
        
        for i, sentence in enumerate(sentences):
            # Check if this sentence starts a new topic
            if self._is_topic_sentence(sentence, persona_type):
                # Save previous cluster if it exists
                if current_cluster and len(" ".join(current_cluster)) > 50:
                    subsection = {
                        "document": document_name,
                        "page_number": page_number,
                        "topic": cluster_topic or self._extract_topic_from_sentences(current_cluster),
                        "raw_content": " ".join(current_cluster),
                        "extraction_method": "sentence_based",
                        "refinement_score": 0.6
                    }
                    subsections.append(subsection)
                
                # Start new cluster
                current_cluster = [sentence]
                cluster_topic = self._extract_topic_from_sentence(sentence)
            else:
                current_cluster.append(sentence)
        
        # Add final cluster
        if current_cluster and len(" ".join(current_cluster)) > 50:
            subsection = {
                "document": document_name,
                "page_number": page_number,
                "topic": cluster_topic or self._extract_topic_from_sentences(current_cluster),
                "raw_content": " ".join(current_cluster),
                "extraction_method": "sentence_based",
                "refinement_score": 0.6
            }
            subsections.append(subsection)
        
        return subsections
    
    def _extract_structured_subsections(self, content: str, persona_type: str, 
                                      document_name: str, page_number: int) -> List[Dict[str, Any]]:
        """Extract subsections from lists and structured content."""
        subsections = []
        
        # Extract bulleted lists
        bullet_pattern = r'([•\-\*]\s*[^•\-\*\n]+(?:\n[^•\-\*\n]+)*)'
        bullet_matches = re.findall(bullet_pattern, content, re.MULTILINE)
        
        for match in bullet_matches:
            if len(match.strip()) > 40:
                subsection = {
                    "document": document_name,
                    "page_number": page_number,
                    "topic": self._extract_topic_from_list(match),
                    "raw_content": match.strip(),
                    "extraction_method": "structured_list",
                    "refinement_score": 0.8  # High score for structured content
                }
                subsections.append(subsection)
        
        # Extract numbered lists
        numbered_pattern = r'(\d+\.?\s*[^\d\n]+(?:\n[^\d\n]+)*)'
        numbered_matches = re.findall(numbered_pattern, content, re.MULTILINE)
        
        for match in numbered_matches:
            if len(match.strip()) > 40 and not re.match(r'^\d+\.\s*$', match.strip()):
                subsection = {
                    "document": document_name,
                    "page_number": page_number,
                    "topic": self._extract_topic_from_list(match),
                    "raw_content": match.strip(),
                    "extraction_method": "numbered_list",
                    "refinement_score": 0.85  # Higher score for numbered content
                }
                subsections.append(subsection)
        
        return subsections
    
    def _is_topic_sentence(self, sentence: str, persona_type: str) -> bool:
        """Determine if a sentence introduces a new topic."""
        sentence_lower = sentence.lower().strip()
        
        # Check for topic indicators
        topic_indicators = [
            "here are", "the following", "these include", "you can", "you should",
            "it is important", "consider", "remember", "note that", "key"
        ]
        
        if any(indicator in sentence_lower for indicator in topic_indicators):
            return True
        
        # Check for persona-specific topic patterns
        if persona_type in self.refinement_patterns:
            keywords = self.refinement_patterns[persona_type]["enhancement_keywords"]
            if any(keyword in sentence_lower for keyword in keywords):
                return True
        
        # Check for sentence structure patterns
        if re.match(r'^[A-Z][^.!?]*[.!?]$', sentence) and len(sentence.split()) < 15:
            return True
        
        return False
    
    def _extract_topic_from_sentence(self, sentence: str) -> str:
        """Extract topic from a sentence."""
        # Simple approach: take first few words
        words = sentence.split()[:5]
        topic = " ".join(words)
        if topic.endswith((":", ".", "!", "?")):
            topic = topic[:-1]
        return topic
    
    def _extract_topic_from_sentences(self, sentences: List[str]) -> str:
        """Extract topic from a group of sentences."""
        if not sentences:
            return "Content Section"
        
        first_sentence = sentences[0]
        return self._extract_topic_from_sentence(first_sentence)
    
    def _extract_topic_from_list(self, list_content: str) -> str:
        """Extract topic from list content."""
        # Take the first meaningful part
        lines = list_content.split('\n')
        if lines:
            first_line = lines[0].strip()
            # Remove list markers
            cleaned = re.sub(r'^[•\-\*\d+\.]\s*', '', first_line)
            words = cleaned.split()[:6]
            return " ".join(words)
        return "List Content"
    
    def _refine_subsection_content(self, subsection: Dict[str, Any], 
                                 persona_type: str, doc_context: Dict[str, Any]) -> Dict[str, Any]:
        """Refine and enhance subsection content."""
        raw_content = subsection["raw_content"]
        
        # Apply refinement rules
        refined_content = raw_content
        
        if self.enhancement_rules["context_expansion"]:
            refined_content = self._expand_context(refined_content, doc_context, persona_type)
        
        if self.enhancement_rules["list_formatting"]:
            refined_content = self._format_lists(refined_content)
        
        if self.enhancement_rules["sentence_completion"]:
            refined_content = self._complete_sentences(refined_content)
        
        if self.enhancement_rules["redundancy_removal"]:
            refined_content = self._remove_redundancy(refined_content)
        
        # Calculate final refinement score
        refinement_score = self._calculate_refinement_score(
            subsection, refined_content, persona_type
        )
        
        return {
            "document": subsection["document"],
            "page_number": subsection["page_number"],
            "refined_text": refined_content,
            "original_content": raw_content,
            "topic": subsection["topic"],
            "extraction_method": subsection["extraction_method"],
            "refinement_score": refinement_score
        }
    
    def _expand_context(self, content: str, doc_context: Dict[str, Any], persona_type: str) -> str:
        """Add relevant context to the content."""
        # For now, return content as-is
        # In a more advanced implementation, we would add relevant context
        return content
    
    def _format_lists(self, content: str) -> str:
        """Format lists and bullet points nicely."""
        # Clean up list formatting
        formatted = re.sub(r'([•\-\*])\s*', r'\1 ', content)
        formatted = re.sub(r'(\d+\.)\s*', r'\1 ', formatted)
        return formatted
    
    def _complete_sentences(self, content: str) -> str:
        """Complete partial sentences where possible."""
        # Basic sentence completion
        sentences = sent_tokenize(content)
        completed_sentences = []
        
        for sentence in sentences:
            if sentence.strip() and not sentence.strip().endswith(('.', '!', '?', ':')):
                sentence = sentence.strip() + '.'
            completed_sentences.append(sentence)
        
        return ' '.join(completed_sentences)
    
    def _remove_redundancy(self, content: str) -> str:
        """Remove redundant information."""
        sentences = sent_tokenize(content)
        unique_sentences = []
        seen_content = set()
        
        for sentence in sentences:
            # Simple deduplication based on first few words
            key = ' '.join(sentence.split()[:5]).lower()
            if key not in seen_content:
                unique_sentences.append(sentence)
                seen_content.add(key)
        
        return ' '.join(unique_sentences)
    
    def _calculate_refinement_score(self, subsection: Dict[str, Any], 
                                  refined_content: str, persona_type: str) -> float:
        """Calculate refinement score for a subsection."""
        base_score = subsection.get("refinement_score", 0.5)
        
        # Content quality factors
        content_length = len(refined_content)
        if 100 <= content_length <= 800:
            base_score += 0.1
        elif content_length > 800:
            base_score -= 0.1
        
        # Information density
        sentences = sent_tokenize(refined_content)
        if len(sentences) >= 3:
            base_score += 0.1
        
        # Persona relevance
        if persona_type in self.refinement_patterns:
            keywords = self.refinement_patterns[persona_type]["enhancement_keywords"]
            matches = sum(1 for keyword in keywords if keyword in refined_content.lower())
            base_score += min(0.2, matches * 0.05)
        
        return min(1.0, base_score)
    
    def _validate_subsection_quality(self, subsection: Dict[str, Any]) -> bool:
        """Validate that subsection meets quality criteria."""
        refined_text = subsection.get("refined_text", "")
        
        # Minimum length requirement
        if len(refined_text) < 50:
            return False
        
        # Maximum length requirement  
        if len(refined_text) > 1500:
            return False
        
        # Must contain meaningful content
        words = word_tokenize(refined_text.lower())
        meaningful_words = [w for w in words if w not in self.stop_words and w.isalpha()]
        if len(meaningful_words) < 10:
            return False
        
        # Must have reasonable refinement score
        if subsection.get("refinement_score", 0) < 0.3:
            return False
        
        return True
    
    def _deduplicate_subsections(self, subsections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate and very similar subsections."""
        unique_subsections = []
        seen_content = set()
        
        for subsection in sorted(subsections, key=lambda x: x.get("refinement_score", 0), reverse=True):
            content = subsection.get("refined_text", "")
            
            # Create a simplified representation for comparison
            words = word_tokenize(content.lower())
            content_key = " ".join(sorted(words[:10]))  # First 10 words, sorted
            
            if content_key not in seen_content:
                unique_subsections.append(subsection)
                seen_content.add(content_key)
        
        return unique_subsections
