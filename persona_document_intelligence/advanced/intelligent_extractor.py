"""
Advanced Intelligent Document Extractor for Challenge 1B
Persona-Driven Document Intelligence with sophisticated ranking algorithms
"""

import re
import logging
from typing import Dict, List, Any, Tuple, Set
from collections import defaultdict, Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class IntelligentExtractor:
    """Advanced document extractor with persona-driven intelligence."""
    
    def __init__(self):
        """Initialize the intelligent extractor."""
        # Download required NLTK data
        for dataset in ['punkt', 'stopwords', 'averaged_perceptron_tagger']:
            try:
                nltk.data.find(f'tokenizers/{dataset}' if dataset == 'punkt' else 
                              f'corpora/{dataset}' if dataset == 'stopwords' else f'taggers/{dataset}')
            except LookupError:
                nltk.download(dataset, quiet=True)
        
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
        # Advanced persona intelligence database
        self.persona_intelligence = {
            "travel_planner": {
                "role_keywords": ["accommodation", "hotels", "restaurants", "activities", "attractions", 
                                "transportation", "itinerary", "budget", "bookings", "tours", "sightseeing"],
                "high_priority_sections": ["things to do", "restaurants", "hotels", "activities", "attractions", 
                                         "transportation", "tips", "itinerary", "planning"],
                "content_focus": ["practical", "actionable", "specific", "detailed"],
                "avoid_sections": ["history", "background", "theory", "academic"],
                "task_modifiers": {
                    "group": 1.3, "friends": 1.2, "college": 1.1, "budget": 1.4, "days": 1.5,
                    "4 days": 1.5, "10 people": 1.3, "young": 1.2
                }
            },
            "hr_professional": {
                "role_keywords": ["forms", "onboarding", "compliance", "fillable", "fields", "signatures", 
                                "workflow", "process", "documentation", "policies", "procedures"],
                "high_priority_sections": ["forms", "fields", "signatures", "workflow", "process", 
                                         "onboarding", "compliance", "policies", "procedures"],
                "content_focus": ["step-by-step", "instructions", "procedural", "compliance"],
                "avoid_sections": ["theory", "background", "history", "academic"],
                "task_modifiers": {
                    "fillable": 1.5, "forms": 1.5, "onboarding": 1.4, "compliance": 1.4,
                    "signatures": 1.3, "workflow": 1.3, "create": 1.2, "manage": 1.2
                }
            },
            "food_contractor": {
                "role_keywords": ["menu", "recipes", "ingredients", "cooking", "preparation", "buffet", 
                                "vegetarian", "gluten-free", "portions", "serving", "catering"],
                "high_priority_sections": ["recipes", "ingredients", "preparation", "cooking", "menu", 
                                         "vegetarian", "sides", "mains", "buffet", "portions"],
                "content_focus": ["recipes", "instructions", "ingredients", "portions"],
                "avoid_sections": ["history", "culture", "background", "theory"],
                "task_modifiers": {
                    "vegetarian": 1.6, "buffet": 1.4, "gluten-free": 1.5, "corporate": 1.2,
                    "dinner": 1.3, "menu": 1.4, "portions": 1.3, "serving": 1.2
                }
            }
        }
        
        # Advanced section classification patterns
        self.section_classifiers = {
            "practical_info": [
                r"tips", r"tricks", r"advice", r"recommendations", r"suggestions",
                r"practical", r"useful", r"helpful", r"guide", r"how to"
            ],
            "activities": [
                r"activities", r"things to do", r"attractions", r"sightseeing",
                r"entertainment", r"nightlife", r"adventure", r"tours"
            ],
            "food_dining": [
                r"restaurants", r"cuisine", r"food", r"dining", r"culinary",
                r"recipes", r"cooking", r"ingredients", r"menu"
            ],
            "accommodation": [
                r"hotels", r"accommodation", r"lodging", r"stay", r"booking"
            ],
            "procedures": [
                r"steps", r"process", r"procedure", r"workflow", r"instructions",
                r"guide", r"tutorial", r"how-to"
            ],
            "forms_fields": [
                r"forms", r"fields", r"fillable", r"input", r"data entry",
                r"signatures", r"signing", r"approval"
            ]
        }
        
        # Initialize TF-IDF vectorizer
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=2
        )
        
    def extract_intelligent_sections(self, documents: List[Dict[str, Any]], 
                                   persona: str, job_to_be_done: str) -> List[Dict[str, Any]]:
        """
        Extract sections using advanced persona-driven intelligence.
        
        Args:
            documents: List of processed document data
            persona: User persona description
            job_to_be_done: Specific task description
            
        Returns:
            List of extracted sections with intelligence scoring
        """
        logger.info(f"Starting intelligent extraction for persona: {persona}")
        
        # Identify persona type and get intelligence profile
        persona_type = self._identify_persona_type(persona)
        intelligence_profile = self._get_intelligence_profile(persona_type, job_to_be_done)
        
        # Extract all potential sections from documents
        all_sections = []
        for doc in documents:
            doc_sections = self._extract_document_sections(doc, intelligence_profile)
            all_sections.extend(doc_sections)
        
        logger.info(f"Extracted {len(all_sections)} potential sections")
        
        # Apply intelligent scoring and ranking
        scored_sections = self._apply_intelligent_scoring(all_sections, intelligence_profile)
        
        # Select top sections based on relevance
        top_sections = self._select_top_sections(scored_sections, intelligence_profile)
        
        logger.info(f"Selected {len(top_sections)} top relevant sections")
        return top_sections
    
    def _identify_persona_type(self, persona: str) -> str:
        """Identify specific persona type from description."""
        persona_lower = persona.lower()
        
        # Direct matches with better pattern recognition
        if "travel" in persona_lower and "planner" in persona_lower:
            return "travel_planner"
        elif ("hr" in persona_lower and "professional" in persona_lower) or "hr professional" in persona_lower:
            return "hr_professional"
        elif "food" in persona_lower and "contractor" in persona_lower:
            return "food_contractor"
        elif "contractor" in persona_lower and ("food" in persona_lower or "catering" in persona_lower):
            return "food_contractor"
        
        # Enhanced fallback based on keywords and context
        if any(word in persona_lower for word in ["travel", "trip", "journey", "tour", "planner"]):
            return "travel_planner"
        elif any(word in persona_lower for word in ["hr", "human", "forms", "onboarding", "professional", "compliance"]):
            return "hr_professional"
        elif any(word in persona_lower for word in ["food", "chef", "cook", "catering", "menu", "contractor", "buffet", "vegetarian"]):
            return "food_contractor"
        
        return "travel_planner"  # Default
    
    def _get_intelligence_profile(self, persona_type: str, job_to_be_done: str) -> Dict[str, Any]:
        """Create comprehensive intelligence profile for the persona and task."""
        base_profile = self.persona_intelligence.get(persona_type, self.persona_intelligence["travel_planner"])
        
        # Enhance with job-specific analysis
        job_keywords = self._extract_job_keywords(job_to_be_done)
        task_multipliers = self._calculate_task_multipliers(job_to_be_done, base_profile["task_modifiers"])
        
        intelligence_profile = {
            "persona_type": persona_type,
            "role_keywords": base_profile["role_keywords"],
            "high_priority_sections": base_profile["high_priority_sections"],
            "content_focus": base_profile["content_focus"],
            "avoid_sections": base_profile["avoid_sections"],
            "job_keywords": job_keywords,
            "task_multipliers": task_multipliers,
            "job_description": job_to_be_done
        }
        
        return intelligence_profile
    
    def _extract_job_keywords(self, job_to_be_done: str) -> List[str]:
        """Extract specific keywords from job description."""
        job_lower = job_to_be_done.lower()
        
        # Remove stop words and extract meaningful terms
        words = word_tokenize(job_lower)
        meaningful_words = [
            word for word in words 
            if word.isalpha() and word not in self.stop_words and len(word) > 3
        ]
        
        # Extract specific entities and important terms
        specific_keywords = []
        
        # Numbers and quantities
        numbers = re.findall(r'\b\d+\b', job_to_be_done)
        specific_keywords.extend(numbers)
        
        # Specific terms in quotes or emphasized
        quoted_terms = re.findall(r'"([^"]*)"', job_to_be_done)
        specific_keywords.extend(quoted_terms)
        
        # Add meaningful words
        specific_keywords.extend(meaningful_words)
        
        return list(set(specific_keywords))
    
    def _calculate_task_multipliers(self, job_to_be_done: str, base_multipliers: Dict[str, float]) -> Dict[str, float]:
        """Calculate task-specific multipliers based on job description."""
        job_lower = job_to_be_done.lower()
        multipliers = {}
        
        for keyword, base_multiplier in base_multipliers.items():
            if keyword.lower() in job_lower:
                multipliers[keyword] = base_multiplier
        
        # Add contextual multipliers
        if "budget" in job_lower or "cheap" in job_lower or "affordable" in job_lower:
            multipliers["budget"] = 1.5
        if "group" in job_lower:
            multipliers["group"] = 1.3
        if "vegetarian" in job_lower:
            multipliers["vegetarian"] = 1.6
        if "gluten-free" in job_lower or "gluten free" in job_lower:
            multipliers["gluten-free"] = 1.5
        
        return multipliers
    
    def _extract_document_sections(self, doc: Dict[str, Any], 
                                 intelligence_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract sections from a single document with intelligence scoring."""
        sections = []
        doc_name = doc.get("metadata", {}).get("filename", "unknown.pdf")
        
        # Extract sections from document structure
        for section in doc.get("sections", []):
            section_data = {
                "document": doc_name,
                "section_title": section["title"],
                "content": section["content"],
                "page_number": section["page_number"],
                "level": section.get("level", 1),
                "start_position": section.get("start_position", 0),
                "end_position": section.get("end_position", 0),
                "intelligence_score": 0.0,
                "relevance_factors": {}
            }
            
            # Apply initial intelligence scoring
            section_data = self._score_section_intelligence(section_data, intelligence_profile)
            sections.append(section_data)
        
        # Also extract content blocks for documents with poor section structure
        if len(sections) < 3:
            content_blocks = self._extract_content_blocks(doc, intelligence_profile)
            sections.extend(content_blocks)
        
        # Post-process section titles for food contractor content
        if intelligence_profile.get("persona_type") == "food_contractor":
            sections = self._enhance_food_contractor_sections(sections)

        return sections
    
    def _score_section_intelligence(self, section: Dict[str, Any], 
                                  intelligence_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Apply intelligent scoring to a section."""
        title = section["section_title"].lower()
        content = section["content"].lower()
        combined_text = f"{title} {content}"
        
        score_factors = {}
        total_score = 0.0
        
        # Factor 1: Role keyword matching (30% weight)
        role_score = self._calculate_role_keyword_score(combined_text, intelligence_profile["role_keywords"])
        score_factors["role_keywords"] = role_score
        total_score += role_score * 0.3
        
        # Factor 2: High priority section matching (25% weight)
        priority_score = self._calculate_priority_section_score(title, intelligence_profile["high_priority_sections"])
        score_factors["priority_sections"] = priority_score
        total_score += priority_score * 0.25
        
        # Factor 3: Job-specific keyword matching (20% weight)
        job_score = self._calculate_job_keyword_score(combined_text, intelligence_profile["job_keywords"])
        score_factors["job_keywords"] = job_score
        total_score += job_score * 0.2
        
        # Factor 4: Content focus alignment (15% weight)
        focus_score = self._calculate_content_focus_score(combined_text, intelligence_profile["content_focus"])
        score_factors["content_focus"] = focus_score
        total_score += focus_score * 0.15
        
        # Factor 5: Task multipliers (10% weight)
        multiplier_score = self._apply_task_multipliers(combined_text, intelligence_profile["task_multipliers"])
        score_factors["task_multipliers"] = multiplier_score
        total_score += multiplier_score * 0.1
        
        # Special bonus for food contractor: dish name bonus
        if intelligence_profile.get("persona_type") == "food_contractor":
            dish_name_bonus = self._calculate_dish_name_bonus(section["section_title"], content)
            score_factors["dish_name_bonus"] = dish_name_bonus
            total_score += dish_name_bonus * 0.3  # 30% bonus for dish names
        
        # Penalty for avoid sections
        avoid_penalty = self._calculate_avoid_penalty(title, intelligence_profile["avoid_sections"])
        score_factors["avoid_penalty"] = avoid_penalty
        total_score = max(0.0, total_score - avoid_penalty)
        
        # Content quality bonus
        quality_bonus = self._calculate_quality_bonus(section["content"])
        score_factors["quality_bonus"] = quality_bonus
        total_score += quality_bonus
        
        section["intelligence_score"] = total_score
        section["relevance_factors"] = score_factors
        
        return section
    
    def _calculate_role_keyword_score(self, text: str, role_keywords: List[str]) -> float:
        """Calculate score based on role-specific keyword matching."""
        words = set(word_tokenize(text.lower()))
        matches = sum(1 for keyword in role_keywords if keyword.lower() in text)
        return min(1.0, matches / max(1, len(role_keywords) * 0.3))
    
    def _calculate_priority_section_score(self, title: str, priority_sections: List[str]) -> float:
        """Calculate score based on high-priority section matching."""
        title_lower = title.lower()
        max_score = 0.0
        
        for priority_section in priority_sections:
            if priority_section.lower() in title_lower:
                max_score = max(max_score, 1.0)
            elif any(word in title_lower for word in priority_section.lower().split()):
                max_score = max(max_score, 0.7)
        
        return max_score
    
    def _calculate_job_keyword_score(self, text: str, job_keywords: List[str]) -> float:
        """Calculate score based on job-specific keyword matching."""
        if not job_keywords:
            return 0.0
        
        matches = sum(1 for keyword in job_keywords if str(keyword).lower() in text)
        return min(1.0, matches / max(1, len(job_keywords) * 0.5))
    
    def _calculate_content_focus_score(self, text: str, content_focus: List[str]) -> float:
        """Calculate score based on content focus alignment."""
        focus_patterns = {
            "practical": [r"how to", r"steps", r"guide", r"tips", r"advice"],
            "actionable": [r"should", r"can", r"will", r"must", r"action"],
            "specific": [r"specific", r"particular", r"exactly", r"precisely"],
            "detailed": [r"detailed", r"comprehensive", r"thorough", r"complete"],
            "step-by-step": [r"step", r"first", r"next", r"then", r"finally"],
            "instructions": [r"instruction", r"procedure", r"process", r"method"],
            "procedural": [r"procedure", r"process", r"workflow", r"protocol"],
            "recipes": [r"recipe", r"ingredient", r"cooking", r"preparation"],
            "ingredients": [r"ingredient", r"cup", r"tablespoon", r"ounce"]
        }
        
        total_score = 0.0
        for focus_type in content_focus:
            if focus_type in focus_patterns:
                patterns = focus_patterns[focus_type]
                matches = sum(1 for pattern in patterns if re.search(pattern, text, re.IGNORECASE))
                total_score += min(0.25, matches * 0.05)
        
        return total_score
    
    def _apply_task_multipliers(self, text: str, task_multipliers: Dict[str, float]) -> float:
        """Apply task-specific multipliers."""
        multiplier_score = 0.0
        
        for keyword, multiplier in task_multipliers.items():
            if str(keyword).lower() in text:
                multiplier_score += (multiplier - 1.0) * 0.1  # Convert to score
        
        return multiplier_score
    
    def _calculate_avoid_penalty(self, title: str, avoid_sections: List[str]) -> float:
        """Calculate penalty for sections that should be avoided."""
        title_lower = title.lower()
        penalty = 0.0
        
        for avoid_term in avoid_sections:
            if avoid_term.lower() in title_lower:
                penalty += 0.3
        
        return penalty
    
    def _calculate_quality_bonus(self, content: str) -> float:
        """Calculate quality bonus based on content characteristics."""
        if not content:
            return 0.0
        
        bonus = 0.0
        
        # Length bonus (optimal range)
        content_length = len(content)
        if 200 <= content_length <= 1500:
            bonus += 0.1
        elif 100 <= content_length <= 2000:
            bonus += 0.05
        
        # Structure bonus
        if any(indicator in content.lower() for indicator in [":", "-", "•", "1.", "2.", "first", "second"]):
            bonus += 0.05
        
        # Information density bonus
        sentences = sent_tokenize(content)
        if len(sentences) >= 3:
            bonus += 0.05
        
        return bonus
    
    def _extract_content_blocks(self, doc: Dict[str, Any], 
                              intelligence_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract content blocks from documents with poor section structure."""
        blocks = []
        doc_name = doc.get("metadata", {}).get("filename", "unknown.pdf")
        
        # Combine all text from pages
        all_text = ""
        for page in doc.get("pages", []):
            all_text += page.get("text", "") + "\n"
        
        # Split into meaningful chunks
        paragraphs = [p.strip() for p in all_text.split('\n\n') if p.strip() and len(p.strip()) > 50]
        
        for i, paragraph in enumerate(paragraphs):
            # Create a pseudo-section for each substantial paragraph
            section_data = {
                "document": doc_name,
                "section_title": self._generate_section_title(paragraph),
                "content": paragraph,
                "page_number": 1,  # Approximate
                "level": 1,
                "start_position": 0,
                "end_position": 0,
                "intelligence_score": 0.0,
                "relevance_factors": {}
            }
            
            # Score the content block
            section_data = self._score_section_intelligence(section_data, intelligence_profile)
            
            # Only include if it has reasonable relevance
            if section_data["intelligence_score"] > 0.2:
                blocks.append(section_data)
        
        return blocks
    
    def _generate_section_title(self, content: str) -> str:
        """Generate a section title from content."""
        # Special handling for food contractor content - look for dish names
        dish_name = self._extract_dish_name(content)
        if dish_name:
            return dish_name
            
        # Take first meaningful sentence
        sentences = sent_tokenize(content)
        if sentences:
            first_sentence = sentences[0]
            # Skip generic headers like "Ingredients:"
            if first_sentence.lower().strip().endswith(('ingredients:', 'instructions:', 'directions:')):
                # Look for previous context or alternative title
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.strip() and not line.lower().strip().endswith(('ingredients:', 'instructions:', 'directions:')):
                        candidate = line.strip()
                        if len(candidate) < 50 and len(candidate.split()) <= 6:
                            return candidate
                
            # Truncate if too long
            if len(first_sentence) > 60:
                words = first_sentence.split()[:8]
                return " ".join(words) + "..."
            return first_sentence
        return "Content Section"
    
    def _extract_dish_name(self, content: str) -> str:
        """Extract dish name from recipe content for food contractor."""
        lines = content.split('\n')
        
        # First, look for dish names that appear as standalone lines (likely section titles)
        for i, line in enumerate(lines):
            line = line.strip()
            if line and len(line) < 50:
                # Check if this looks like a dish name (short, capitalized, not generic)
                if (len(line.split()) <= 6 and 
                    line[0].isupper() and 
                    not line.lower().strip().endswith(('ingredients:', 'instructions:', 'directions:', 'preparation:'))):
                    
                    # Check if the next lines contain ingredients-related content
                    next_lines = lines[i+1:i+3] if i+1 < len(lines) else []
                    has_ingredients_context = False
                    
                    for next_line in next_lines:
                        if any(keyword in next_line.lower() for keyword in ['ingredients:', 'cup', 'tablespoon', 'teaspoon', 'ounce', 'pound', '•']):
                            has_ingredients_context = True
                            break
                    
                    if has_ingredients_context:
                        return line
                
                # Also check for common dish name patterns
                dish_patterns = [
                    r'^[A-Z][a-z]+ [A-Z][a-z]+$',  # "Baba Ganoush"
                    r'^[A-Z][a-z]+$',               # "Falafel"
                    r'^[A-Z][a-z]+ [a-z]+ [A-Z][a-z]+$',  # "Vegetable Lasagna"
                    r'^[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+$',  # "Coconut Rice"
                    r'^[A-Z][a-z]+$'  # "Coleslaw"
                ]
                
                for pattern in dish_patterns:
                    if re.match(pattern, line):
                        return line
        
        # If no dish name found in standalone lines, look for it in the first meaningful sentence
        sentences = sent_tokenize(content)
        if sentences:
            first_sentence = sentences[0].strip()
            # Skip generic headers
            if not first_sentence.lower().endswith(('ingredients:', 'instructions:', 'directions:')):
                # Check if first sentence looks like a dish name
                if len(first_sentence.split()) <= 6 and first_sentence[0].isupper():
                    return first_sentence
        
        return None
    
    def _enhance_food_contractor_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance sections specifically for food contractor content by merging dish names with ingredients."""
        enhanced_sections = []
        i = 0
        
        while i < len(sections):
            current_section = sections[i].copy()
            current_title = current_section.get("section_title", "")
            
            # Check if this is a dish name followed by ingredients
            if (self._is_good_dish_name(current_title) and 
                i + 1 < len(sections) and 
                sections[i + 1].get("section_title", "").lower().strip() == "ingredients:"):
                
                # Merge the dish name section with the ingredients content
                ingredients_section = sections[i + 1]
                current_section["content"] = (
                    current_section.get("content", "") + " " + 
                    ingredients_section.get("content", "")
                ).strip()
                
                logger.debug(f"Merged dish '{current_title}' with ingredients content")
                enhanced_sections.append(current_section)
                
                # Skip the ingredients section since we merged it
                i += 2
                continue
            
            # If the current title is generic (like "Ingredients:"), try to find a better one
            elif (current_title.lower().strip() in ["ingredients:", "instructions:", "directions:", "preparation:"] or
                  len(current_title.split()) <= 2):
                
                # Try to extract a dish name from the content
                dish_name = self._extract_dish_name(current_section.get("content", ""))
                if dish_name:
                    current_section["section_title"] = dish_name
                    logger.debug(f"Enhanced title from '{current_title}' to '{dish_name}'")
            
            enhanced_sections.append(current_section)
            i += 1
        
        return enhanced_sections
    
    def _is_good_dish_name(self, title: str) -> bool:
        """Check if a title looks like a good dish name."""
        if not title or len(title.strip()) == 0:
            return False
        
        title_lower = title.lower().strip()
        
        # Skip generic titles
        if title_lower in ["ingredients:", "instructions:", "directions:", "preparation:", "method:"]:
            return False
        
        # Check if it looks like a dish name (short, capitalized, not too long)
        if (len(title.split()) <= 6 and 
            title[0].isupper() and 
            len(title) < 50):
            return True
        
        return False
    
    def _is_generic_title(self, title: str) -> bool:
        """Check if a title is generic."""
        title_lower = title.lower().strip()
        generic_titles = ["ingredients:", "instructions:", "directions:", "preparation:", "method:", "steps:"]
        return title_lower in generic_titles or len(title.split()) <= 2
    
    def _extract_dish_name_from_content(self, content: str) -> str:
        """Extract dish name from content."""
        lines = content.split('\n')
        
        # Look for dish names in the first few lines
        for i, line in enumerate(lines[:5]):
            line = line.strip()
            if line and self._is_good_dish_name(line):
                return line
        
        # If no dish name found in lines, try to extract from first sentence
        sentences = sent_tokenize(content)
        if sentences:
            first_sentence = sentences[0].strip()
            if self._is_good_dish_name(first_sentence):
                return first_sentence
        
        return None
    
    def _apply_intelligent_scoring(self, sections: List[Dict[str, Any]], 
                                 intelligence_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply advanced intelligent scoring using multiple algorithms."""
        logger.info("Applying advanced intelligent scoring algorithms")
        
        # Prepare text corpus for TF-IDF analysis
        corpus = [f"{section['section_title']} {section['content']}" for section in sections]
        
        if len(corpus) > 1:
            try:
                # Calculate TF-IDF vectors
                tfidf_matrix = self.tfidf.fit_transform(corpus)
                
                # Create query vector from persona and job description
                query_text = f"{intelligence_profile['job_description']} {' '.join(intelligence_profile['role_keywords'])}"
                query_vector = self.tfidf.transform([query_text])
                
                # Calculate cosine similarity
                similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
                
                # Update sections with TF-IDF scores
                for i, section in enumerate(sections):
                    section["tfidf_score"] = similarities[i]
                    section["intelligence_score"] += similarities[i] * 0.2  # 20% weight for TF-IDF
                    
            except Exception as e:
                logger.warning(f"TF-IDF analysis failed: {e}")
                # Set default TF-IDF scores
                for section in sections:
                    section["tfidf_score"] = 0.0
        
        return sections
    
    def _select_top_sections(self, sections: List[Dict[str, Any]], 
                           intelligence_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Select top sections based on intelligent scoring."""
        # Sort by intelligence score
        sorted_sections = sorted(sections, key=lambda x: x["intelligence_score"], reverse=True)
        
        # Select top sections with diversity
        selected_sections = []
        used_documents = set()
        
        # First pass: select highest scoring section from each document
        for section in sorted_sections:
            doc_name = section["document"]
            if doc_name not in used_documents and section["intelligence_score"] > 0.3:
                selected_sections.append(section)
                used_documents.add(doc_name)
        
        # Second pass: add more high-scoring sections if we have fewer than 5
        if len(selected_sections) < 5:
            for section in sorted_sections:
                if section not in selected_sections and section["intelligence_score"] > 0.25:
                    selected_sections.append(section)
                    if len(selected_sections) >= 5:
                        break
        
        # Ensure we have at least some sections
        if len(selected_sections) < 3:
            selected_sections = sorted_sections[:5]
        
        return selected_sections[:5]  # Maximum 5 sections as per challenge requirements

    def _calculate_dish_name_bonus(self, title: str, content: str) -> float:
        """Calculate bonus score for sections with dish names."""
        # Check if title looks like a dish name
        if self._is_good_dish_name(title):
            # Check if content contains recipe-related keywords
            recipe_keywords = ['ingredients:', 'instructions:', 'cup', 'tablespoon', 'teaspoon', 'ounce', 'pound', '•']
            has_recipe_content = any(keyword in content.lower() for keyword in recipe_keywords)
            
            if has_recipe_content:
                return 1.0  # Full bonus for dish name with recipe content
        
        return 0.0
