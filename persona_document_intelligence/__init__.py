"""
Persona-Driven Document Intelligence System
Round 1B: Connect What Matters — For the User Who Matters

A system for extracting and prioritizing relevant sections from documents
based on a specific persona and their job-to-be-done.
"""

from persona_document_intelligence.core.document_processor import DocumentProcessor
from persona_document_intelligence.core.persona_analyzer import PersonaAnalyzer
from persona_document_intelligence.core.section_extractor import SectionExtractor
from persona_document_intelligence.core.relevance_ranker import RelevanceRanker
from persona_document_intelligence.utils.output_formatter import OutputFormatter

# Advanced AI components for sophisticated persona intelligence
from persona_document_intelligence.advanced.intelligent_extractor import IntelligentExtractor
from persona_document_intelligence.advanced.subsection_analyzer import AdvancedSubsectionAnalyzer

__all__ = [
    'DocumentProcessor',
    'PersonaAnalyzer', 
    'SectionExtractor',
    'RelevanceRanker',
    'OutputFormatter',
    'IntelligentExtractor',
    'AdvancedSubsectionAnalyzer',
] 