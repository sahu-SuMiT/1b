"""
Persona analysis module for understanding user roles and job requirements.
"""

import logging
import re
from typing import Dict, List, Any, Set
from collections import defaultdict

logger = logging.getLogger(__name__)

class PersonaAnalyzer:
    """Analyzes persona and job requirements to extract relevant criteria."""
    
    def __init__(self):
        """Initialize the persona analyzer."""
        # Domain-specific keywords and their relevance weights
        self.domain_keywords = {
            "academic": {
                "research": 0.9, "methodology": 0.8, "literature": 0.7, "analysis": 0.8,
                "findings": 0.7, "conclusion": 0.6, "references": 0.5, "abstract": 0.6,
                "introduction": 0.5, "background": 0.6, "results": 0.8, "discussion": 0.7
            },
            "business": {
                "revenue": 0.9, "profit": 0.8, "market": 0.8, "strategy": 0.9,
                "financial": 0.9, "investment": 0.8, "growth": 0.7, "performance": 0.8,
                "analysis": 0.7, "trends": 0.8, "competition": 0.7, "forecast": 0.7
            },
            "education": {
                "concept": 0.8, "theory": 0.7, "principle": 0.8, "mechanism": 0.9,
                "reaction": 0.9, "formula": 0.8, "equation": 0.8, "definition": 0.7,
                "example": 0.6, "application": 0.7, "practice": 0.6, "review": 0.6
            },
            "technical": {
                "algorithm": 0.9, "implementation": 0.8, "architecture": 0.8, "design": 0.7,
                "performance": 0.8, "optimization": 0.8, "framework": 0.7, "protocol": 0.8,
                "specification": 0.7, "documentation": 0.6, "testing": 0.7, "deployment": 0.7
            },
            "tourism": {
                "travel": 0.9, "destinations": 0.9, "attractions": 0.8, "activities": 0.8,
                "accommodation": 0.8, "hotels": 0.8, "restaurants": 0.8, "dining": 0.7,
                "sightseeing": 0.8, "tours": 0.7, "beaches": 0.8, "culture": 0.7,
                "entertainment": 0.7, "nightlife": 0.7, "transportation": 0.8, "budget": 0.8,
                "itinerary": 0.9, "tips": 0.7, "advice": 0.6, "vacation": 0.8, "trip": 0.9
            },
            "logistics": {
                "planning": 0.9, "schedule": 0.8, "coordination": 0.8, "organization": 0.8,
                "transportation": 0.9, "booking": 0.8, "arrangements": 0.8, "logistics": 0.9,
                "timing": 0.7, "routes": 0.8, "reservations": 0.8, "management": 0.7
            }
        }
        
        # Persona-specific focus areas
        self.persona_focus = {
            "researcher": {
                "domains": ["academic"],
                "keywords": ["methodology", "findings", "analysis", "conclusion", "literature"],
                "section_types": ["methodology", "results", "discussion", "conclusion", "related work"]
            },
            "student": {
                "domains": ["education", "academic"],
                "keywords": ["concept", "theory", "principle", "example", "practice"],
                "section_types": ["introduction", "concepts", "examples", "summary", "exercises"]
            },
            "analyst": {
                "domains": ["business", "technical"],
                "keywords": ["analysis", "trends", "performance", "strategy", "forecast"],
                "section_types": ["analysis", "results", "trends", "recommendations", "summary"]
            },
            "investor": {
                "domains": ["business"],
                "keywords": ["revenue", "profit", "growth", "financial", "market"],
                "section_types": ["financial", "performance", "outlook", "risks", "opportunities"]
            },
            "entrepreneur": {
                "domains": ["business", "technical"],
                "keywords": ["strategy", "market", "growth", "innovation", "opportunity"],
                "section_types": ["strategy", "market analysis", "business model", "growth", "competition"]
            },
            "travel_planner": {
                "domains": ["tourism", "logistics"],
                "keywords": ["destinations", "activities", "accommodation", "transportation", "attractions", "restaurants", "budget", "itinerary", "sightseeing", "tours", "hotels", "flights", "dining", "entertainment", "nightlife", "beaches", "culture", "history", "tips", "advice"],
                "section_types": ["destinations", "activities", "accommodation", "dining", "transportation", "attractions", "entertainment", "tips", "culture", "budget"]
            },
            "hr_professional": {
                "domains": ["business", "technical"],
                "keywords": ["forms", "fillable", "onboarding", "compliance", "workflow", "process", "signature", "approval", "fields", "documents", "management", "automation", "creation", "templates", "requirements"],
                "section_types": ["forms", "workflow", "processes", "compliance", "documentation", "management", "creation", "automation", "templates", "requirements"]
            },
            "food_contractor": {
                "domains": ["business"],
                "keywords": ["menu", "vegetarian", "gluten-free", "buffet", "dinner", "corporate", "catering", "recipes", "ingredients", "dietary", "preparation", "cooking", "food", "nutrition", "planning"],
                "section_types": ["menu", "recipes", "ingredients", "preparation", "dietary", "nutrition", "planning", "catering", "cooking", "food"]
            },
            "journalist": {
                "domains": ["academic", "business"],
                "keywords": ["summary", "key points", "highlights", "trends", "impact"],
                "section_types": ["summary", "highlights", "key findings", "implications", "background"]
            }
        }
        
    def analyze_persona(self, persona: str, job_to_be_done: str) -> Dict[str, Any]:
        """
        Analyze persona and job requirements to extract relevant criteria.
        
        Args:
            persona: Description of the user role
            job_to_be_done: Specific task to be accomplished
            
        Returns:
            Dictionary containing analysis results and relevance criteria
        """
        logger.info(f"Analyzing persona: {persona[:50]}...")
        logger.info(f"Job to be done: {job_to_be_done[:100]}...")
        
        # Extract persona type
        persona_type = self._identify_persona_type(persona)
        
        # Extract job requirements
        job_requirements = self._extract_job_requirements(job_to_be_done)
        
        # Combine persona and job analysis
        analysis = {
            "persona_type": persona_type,
            "persona_description": persona,
            "job_description": job_to_be_done,
            "focus_domains": self._get_focus_domains(persona_type),
            "relevant_keywords": self._get_relevant_keywords(persona_type, job_requirements),
            "section_priorities": self._get_section_priorities(persona_type, job_requirements),
            "content_filters": self._get_content_filters(persona_type, job_requirements),
            "relevance_criteria": self._build_relevance_criteria(persona_type, job_requirements)
        }
        
        logger.info(f"Identified persona type: {persona_type}")
        logger.info(f"Focus domains: {analysis['focus_domains']}")
        logger.info(f"Relevant keywords: {len(analysis['relevant_keywords'])} keywords")
        
        return analysis
    
    def _identify_persona_type(self, persona: str) -> str:
        """Identify the primary persona type from the description."""
        persona_lower = persona.lower()
        
        # Direct matches for Challenge 1B personas
        if "travel" in persona_lower and "planner" in persona_lower:
            return "travel_planner"
        elif ("hr" in persona_lower and "professional" in persona_lower) or "hr professional" in persona_lower:
            return "hr_professional"
        elif "food" in persona_lower and "contractor" in persona_lower:
            return "food_contractor"
        elif "contractor" in persona_lower and ("food" in persona_lower or "catering" in persona_lower):
            return "food_contractor"
        
        # Check for existing persona types in keys
        for persona_type in self.persona_focus.keys():
            if persona_type in persona_lower.replace(" ", "_") or persona_type.replace("_", " ") in persona_lower:
                return persona_type
        
        # Enhanced fallback based on keywords and context
        if any(word in persona_lower for word in ["travel", "trip", "journey", "tour", "planner"]):
            return "travel_planner"
        elif any(word in persona_lower for word in ["hr", "human", "forms", "onboarding", "professional", "compliance"]):
            return "hr_professional"
        elif any(word in persona_lower for word in ["food", "chef", "cook", "catering", "menu", "contractor", "buffet", "vegetarian"]):
            return "food_contractor"
        
        # Check for academic indicators
        if any(word in persona.lower() for word in ["phd", "researcher", "professor", "academic", "scholar"]):
            return "researcher"
        
        # Check for student indicators
        if any(word in persona.lower() for word in ["student", "undergraduate", "graduate", "learner"]):
            return "student"
        
        # Check for business indicators
        if any(word in persona_lower for word in ["analyst", "investment", "financial", "business"]):
            return "analyst"
        
        # Default to travel_planner for Challenge 1B
        return "travel_planner"
    
    def _extract_job_requirements(self, job_to_be_done: str) -> Dict[str, Any]:
        """Extract specific requirements from the job description."""
        job_lower = job_to_be_done.lower()
        
        requirements = {
            "action_type": self._identify_action_type(job_lower),
            "content_type": self._identify_content_type(job_lower),
            "scope": self._identify_scope(job_lower),
            "specific_keywords": self._extract_specific_keywords(job_lower)
        }
        
        return requirements
    
    def _identify_action_type(self, job_text: str) -> str:
        """Identify the type of action required."""
        if any(word in job_text for word in ["analyze", "analysis", "examine", "study"]):
            return "analysis"
        elif any(word in job_text for word in ["summarize", "review", "overview", "summary"]):
            return "summary"
        elif any(word in job_text for word in ["prepare", "create", "develop", "build"]):
            return "creation"
        elif any(word in job_text for word in ["identify", "find", "locate", "discover"]):
            return "identification"
        else:
            return "general"
    
    def _identify_content_type(self, job_text: str) -> str:
        """Identify the type of content needed."""
        if any(word in job_text for word in ["literature", "research", "papers", "studies"]):
            return "academic"
        elif any(word in job_text for word in ["financial", "revenue", "profit", "market"]):
            return "business"
        elif any(word in job_text for word in ["concept", "theory", "mechanism", "principle"]):
            return "educational"
        elif any(word in job_text for word in ["method", "approach", "technique", "algorithm"]):
            return "technical"
        else:
            return "general"
    
    def _identify_scope(self, job_text: str) -> str:
        """Identify the scope of the task."""
        if any(word in job_text for word in ["comprehensive", "complete", "full", "detailed"]):
            return "comprehensive"
        elif any(word in job_text for word in ["overview", "summary", "brief", "key"]):
            return "overview"
        elif any(word in job_text for word in ["specific", "particular", "focused", "targeted"]):
            return "focused"
        else:
            return "general"
    
    def _extract_specific_keywords(self, job_text: str) -> Set[str]:
        """Extract specific keywords mentioned in the job description."""
        # Remove common words and extract meaningful terms
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        words = re.findall(r'\b\w+\b', job_text.lower())
        specific_words = {word for word in words if word not in common_words and len(word) > 3}
        return specific_words
    
    def _get_focus_domains(self, persona_type: str) -> List[str]:
        """Get the focus domains for a persona type."""
        if persona_type in self.persona_focus:
            return self.persona_focus[persona_type]["domains"]
        return ["general"]
    
    def _get_relevant_keywords(self, persona_type: str, job_requirements: Dict[str, Any]) -> Dict[str, float]:
        """Get relevant keywords with their weights."""
        keywords = {}
        
        # Add persona-specific keywords
        if persona_type in self.persona_focus:
            for keyword in self.persona_focus[persona_type]["keywords"]:
                keywords[keyword] = 0.8
        
        # Add domain-specific keywords
        for domain in self._get_focus_domains(persona_type):
            if domain in self.domain_keywords:
                keywords.update(self.domain_keywords[domain])
        
        # Add job-specific keywords
        for keyword in job_requirements["specific_keywords"]:
            keywords[keyword] = 0.9  # High weight for job-specific terms
        
        return keywords
    
    def _get_section_priorities(self, persona_type: str, job_requirements: Dict[str, Any]) -> Dict[str, float]:
        """Get section type priorities based on persona and job."""
        priorities = {}
        
        # Base priorities from persona
        if persona_type in self.persona_focus:
            for section_type in self.persona_focus[persona_type]["section_types"]:
                priorities[section_type] = 0.8
        
        # Adjust based on job requirements
        action_type = job_requirements["action_type"]
        if action_type == "analysis":
            priorities.update({
                "methodology": 0.9,
                "results": 0.9,
                "analysis": 0.9,
                "discussion": 0.8
            })
        elif action_type == "summary":
            priorities.update({
                "summary": 0.9,
                "conclusion": 0.8,
                "key findings": 0.9,
                "overview": 0.8
            })
        
        return priorities
    
    def _get_content_filters(self, persona_type: str, job_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Get content filtering criteria."""
        filters = {
            "min_section_length": 50,  # Minimum characters for a section
            "max_section_length": 2000,  # Maximum characters for a section
            "preferred_section_types": self._get_section_priorities(persona_type, job_requirements),
            "exclude_patterns": [
                r"^table of contents$",
                r"^references?$",
                r"^bibliography$",
                r"^appendix",
                r"^index$"
            ]
        }
        
        return filters
    
    def _build_relevance_criteria(self, persona_type: str, job_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive relevance criteria."""
        return {
            "keyword_matching": {
                "weight": 0.4,
                "keywords": self._get_relevant_keywords(persona_type, job_requirements)
            },
            "section_relevance": {
                "weight": 0.3,
                "priorities": self._get_section_priorities(persona_type, job_requirements)
            },
            "content_quality": {
                "weight": 0.2,
                "min_length": 50,
                "max_length": 2000
            },
            "job_specificity": {
                "weight": 0.1,
                "specific_keywords": job_requirements["specific_keywords"]
            }
        } 