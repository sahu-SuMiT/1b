#!/usr/bin/env python3
"""
Advanced Persona-Driven Document Intelligence System for Challenge 1B
Adobe Hackathon - Round 1B: Connect What Matters — For the User Who Matters

This system uses sophisticated AI to extract and prioritize document sections
based on specific personas and their job-to-be-done requirements.
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime

# Advanced AI components for Challenge 1B
from persona_document_intelligence.core.document_processor import DocumentProcessor
from persona_document_intelligence.core.persona_analyzer import PersonaAnalyzer
from persona_document_intelligence.advanced.intelligent_extractor import IntelligentExtractor
from persona_document_intelligence.advanced.subsection_analyzer import AdvancedSubsectionAnalyzer
from persona_document_intelligence.utils.output_formatter import OutputFormatter


def find_pdf_directory(base_dir, documents_info):
    """
    Intelligently find the directory containing PDF files with comprehensive fallback.
    
    Priority order:
    1. Same directory as input file (base_dir/)
    2. PDFs subdirectory (base_dir/PDFs/)
    3. docs subdirectory (base_dir/docs/)
    4. documents subdirectory (base_dir/documents/)
    5. Any subdirectory containing PDF files
    6. Search in parent directories
    7. Search in common locations
    
    Args:
        base_dir (str): Base directory to start searching from
        documents_info (list): List of document info with filenames
        
    Returns:
        str: Path to directory containing PDFs
    """
    if not documents_info:
        logging.warning("⚠️  No documents specified in input")
        return base_dir
    
    # Extract just the filenames for easier searching
    filenames = [doc.get('filename', '') for doc in documents_info if doc.get('filename')]
    
    if not filenames:
        logging.warning("⚠️  No valid filenames found in documents")
        return base_dir
    
    logging.info(f"🔍 Searching for {len(filenames)} PDF files...")
    
    # Priority 1: Check same directory as input file
    found_count = count_files_in_directory(base_dir, filenames)
    if found_count > 0:
        logging.info(f"📁 Found {found_count}/{len(filenames)} PDFs in input directory: {base_dir}")
        if found_count == len(filenames):
            return base_dir
    
    # Priority 2-4: Check common subdirectory names
    common_subdirs = ["PDFs", "pdfs", "docs", "documents", "files", "pdf"]
    
    for subdir_name in common_subdirs:
        subdir_path = os.path.join(base_dir, subdir_name)
        if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
            found_count = count_files_in_directory(subdir_path, filenames)
            if found_count > 0:
                logging.info(f"📁 Found {found_count}/{len(filenames)} PDFs in {subdir_name}/: {subdir_path}")
                if found_count == len(filenames):
                    return subdir_path
    
    # Priority 5: Search all subdirectories for PDF files
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item not in common_subdirs:  # Skip already checked
            found_count = count_files_in_directory(item_path, filenames)
            if found_count > 0:
                logging.info(f"📁 Found {found_count}/{len(filenames)} PDFs in {item}/: {item_path}")
                if found_count == len(filenames):
                    return item_path
    
    # Priority 6: Check parent directory
    parent_dir = os.path.dirname(base_dir)
    if parent_dir != base_dir:  # Avoid infinite loop
        found_count = count_files_in_directory(parent_dir, filenames)
        if found_count > 0:
            logging.info(f"📁 Found {found_count}/{len(filenames)} PDFs in parent directory: {parent_dir}")
            if found_count == len(filenames):
                return parent_dir
    
    # Priority 7: Search common absolute locations
    common_locations = [
        "input",
        "input/PDFs", 
        "input/docs",
        "PDFs",
        "docs",
        "documents"
    ]
    
    for location in common_locations:
        if os.path.exists(location) and os.path.isdir(location):
            found_count = count_files_in_directory(location, filenames)
            if found_count > 0:
                logging.info(f"📁 Found {found_count}/{len(filenames)} PDFs in {location}/")
                if found_count == len(filenames):
                    return location
    
    # Fallback: Return the directory with the most matches found
    best_dir = base_dir
    best_count = count_files_in_directory(base_dir, filenames)
    
    # Check all previously searched locations and return the one with most files
    all_search_paths = [
        base_dir,
        *[os.path.join(base_dir, subdir) for subdir in common_subdirs 
          if os.path.exists(os.path.join(base_dir, subdir))],
        *[os.path.join(base_dir, item) for item in os.listdir(base_dir)
          if os.path.isdir(os.path.join(base_dir, item))],
        parent_dir if parent_dir != base_dir else None,
        *[loc for loc in common_locations if os.path.exists(loc)]
    ]
    
    for search_path in all_search_paths:
        if search_path and os.path.exists(search_path):
            count = count_files_in_directory(search_path, filenames)
            if count > best_count:
                best_count = count
                best_dir = search_path
    
    if best_count == 0:
        logging.warning(f"⚠️  No PDF files found! Searched {len(all_search_paths)} locations")
        logging.warning(f"   Looking for: {', '.join(filenames[:3])}{'...' if len(filenames) > 3 else ''}")
    else:
        logging.info(f"📁 Best match: {best_dir} (found {best_count}/{len(filenames)} files)")
    
    return best_dir


def count_files_in_directory(directory, filenames):
    """Count how many of the specified filenames exist in the given directory."""
    if not os.path.exists(directory) or not os.path.isdir(directory):
        return 0
    
    count = 0
    for filename in filenames:
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            count += 1
    
    return count


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('challenge1b_processing.log', mode='w', encoding='utf-8')
        ]
    )


def main():
    """Main entry point for the Advanced Persona-Driven Document Intelligence System."""
    setup_logging()
    
    parser = argparse.ArgumentParser(description='Advanced Persona-Driven Document Intelligence for Challenge 1B')
    parser.add_argument('command', choices=['process', 'test'], 
                       help='Command: process (single input) or test (all collections)')
    parser.add_argument('--input', '-i', help='Path to input configuration file')
    parser.add_argument('--output', '-o', help='Path to output file')
    parser.add_argument('--collection', '-c', help='Specific collection to test (1, 2, or 3)')
    
    args = parser.parse_args()
    
    logging.info("🚀 Starting Advanced Persona-Driven Document Intelligence System")
    logging.info("="*70)
    
    try:
        if args.command == 'test':
            success = test_all_collections(args.collection)
            if success:
                logging.info("✅ All tests completed successfully!")
                sys.exit(0)
            else:
                logging.error("❌ Some tests failed!")
                sys.exit(1)
        else:
            # Single processing mode
            input_path = args.input or find_input_file()
            if not input_path:
                logging.error("❌ No input configuration file found or specified")
                sys.exit(1)
            
            output_path = args.output or "output/challenge1b_output.json"
            success = process_single_input(input_path, output_path)
            
            if success:
                logging.info("✅ Processing completed successfully!")
                sys.exit(0)
            else:
                logging.error("❌ Processing failed!")
                sys.exit(1)
                
    except KeyboardInterrupt:
        logging.info("⏹️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)


def find_input_file():
    """Find input configuration file in standard locations."""
    possible_paths = [
        "input/config.json",
        "input/challenge1b_input.json",
        "input/input.json"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def process_single_input(input_file, output_file):
    """Process a single input configuration file."""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Handle different config formats
        if 'persona' in config and isinstance(config['persona'], dict):
            persona = config['persona'].get('role', 'Unknown')
        else:
            persona = config.get('persona', 'Unknown')
            
        if 'job_to_be_done' in config and isinstance(config['job_to_be_done'], dict):
            job_to_be_done = config['job_to_be_done'].get('task', 'Unknown task')
        else:
            job_to_be_done = config.get('job_to_be_done', 'Unknown task')
        
        documents_info = config.get('documents', [])
        
        # Advanced PDF directory detection with comprehensive fallback
        input_dir = os.path.dirname(input_file)
        pdfs_dir = find_pdf_directory(input_dir, documents_info)
        
        return process_collection(input_file, pdfs_dir, output_file)
        
    except Exception as e:
        logging.error(f"❌ Error processing single input: {e}")
        logging.error(traceback.format_exc())
        return False


def test_all_collections(specific_collection=None):
    """Test the system against all Challenge 1B collections."""
    base_path = "desired_outputs/Challenge_1b"
    
    # Check if base path exists
    if not os.path.exists(base_path):
        logging.error(f"❌ Test data directory not found: {base_path}")
        return False
    
    collections = []
    if specific_collection:
        collections = [f"Collection {specific_collection}"]
    else:
        collections = [d for d in os.listdir(base_path) 
                      if d.startswith("Collection") and os.path.isdir(os.path.join(base_path, d))]
    
    all_passed = True
    
    for collection in sorted(collections):
        collection_path = os.path.join(base_path, collection)
        logging.info(f"🧪 Testing {collection}...")
        
        # Find input and expected output files
        input_file = os.path.join(collection_path, "challenge1b_input.json")
        expected_output_file = os.path.join(collection_path, "challenge1b_output.json")
        pdfs_dir = os.path.join(collection_path, "PDFs")
        
        if not os.path.exists(input_file):
            logging.error(f"❌ Input file not found: {input_file}")
            all_passed = False
            continue
            
        if not os.path.exists(expected_output_file):
            logging.error(f"❌ Expected output file not found: {expected_output_file}")
            all_passed = False
            continue
            
        if not os.path.exists(pdfs_dir):
            logging.error(f"❌ PDFs directory not found: {pdfs_dir}")
            all_passed = False
            continue
        
        # Process this collection
        output_file = f"output/test_{collection.lower().replace(' ', '_')}_output.json"
        success = process_collection(input_file, pdfs_dir, output_file)
        
        if not success:
            logging.error(f"❌ Failed to process {collection}")
            all_passed = False
            continue
            
        # Compare with expected output
        logging.info(f"📊 Comparing results for {collection}...")
        comparison_result = compare_outputs(output_file, expected_output_file)
        
        if comparison_result:
            logging.info(f"✅ {collection} passed validation!")
        else:
            logging.warning(f"⚠️  {collection} has differences from expected output")
            all_passed = False
    
    return all_passed


def process_collection(input_file, pdfs_dir, output_file):
    """Process a single collection with its PDFs."""
    try:
        # Load input configuration
        with open(input_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Extract persona and job information
        persona = config.get('persona', {}).get('role', 'Unknown')
        job_to_be_done = config.get('job_to_be_done', {}).get('task', 'Unknown task')
        documents_info = config.get('documents', [])
        
        logging.info(f"👤 Persona: {persona}")
        logging.info(f"🎯 Job to be done: {job_to_be_done}")
        logging.info(f"📄 Processing {len(documents_info)} documents...")
        
        # Initialize AI components
        logging.info("📋 Initializing advanced AI components...")
        document_processor = DocumentProcessor()
        persona_analyzer = PersonaAnalyzer()
        intelligent_extractor = IntelligentExtractor()
        subsection_analyzer = AdvancedSubsectionAnalyzer()
        output_formatter = OutputFormatter()
        
        # Process all documents
        documents = []
        for doc_info in documents_info:
            pdf_path = os.path.join(pdfs_dir, doc_info['filename'])
            if os.path.exists(pdf_path):
                logging.info(f"📖 Processing: {doc_info['filename']}")
                # Use the correct method name and handle Path object
                doc_content = document_processor.process_document(Path(pdf_path))
                
                # Debug: Check what sections were extracted
                sections_count = len(doc_content.get('sections', []))
                logging.info(f"   📋 Found {sections_count} sections in {doc_info['filename']}")
                
                documents.append({
                    'filename': doc_info['filename'],
                    'title': doc_info.get('title', doc_info['filename']),
                    'content': doc_content
                })
            else:
                logging.warning(f"⚠️  PDF not found: {pdf_path}")
        
        if not documents:
            logging.error("❌ No documents were successfully processed")
            return False
        
        # Analyze persona and extract relevant sections
        logging.info("🧠 Analyzing persona and extracting relevant sections...")
        persona_profile = persona_analyzer.analyze_persona(persona, job_to_be_done)
        
        # Use intelligent extractor for section extraction
        # Note: IntelligentExtractor expects the raw document data, not wrapped documents
        raw_documents = [doc['content'] for doc in documents]
        extracted_sections = intelligent_extractor.extract_intelligent_sections(
            raw_documents, persona, job_to_be_done
        )
        
        # Perform subsection analysis
        logging.info("🔍 Performing advanced subsection analysis...")
        subsection_analysis = []
        if extracted_sections:
            # Pass the top sections to the subsection analyzer
            analysis = subsection_analyzer.analyze_subsections(
                extracted_sections[:5], persona_profile.get("persona_type", "travel_planner"), raw_documents
            )
            if analysis:
                subsection_analysis = analysis
        
        # Format output
        logging.info("📝 Formatting output...")
        result = output_formatter.format_output(
            documents=[doc['filename'] for doc in documents],
            persona=persona,
            job_to_be_done=job_to_be_done,
            ranked_sections=extracted_sections,
            processing_timestamp=time.time(),
            subsection_analysis=subsection_analysis
        )
        
        # Save output
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logging.info(f"💾 Output saved to: {output_file}")
        logging.info(f"📊 Extracted {len(extracted_sections)} sections and {len(subsection_analysis)} subsections")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Error processing collection: {e}")
        logging.error(traceback.format_exc())
        return False


def compare_outputs(actual_file, expected_file):
    """Compare actual output with expected output."""
    try:
        with open(actual_file, 'r', encoding='utf-8') as f:
            actual = json.load(f)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            expected = json.load(f)
        
        # Compare structure and key elements
        score = 0
        total_checks = 0
        
        # Check metadata
        if 'metadata' in actual and 'metadata' in expected:
            total_checks += 3
            if actual['metadata'].get('persona') == expected['metadata'].get('persona'):
                score += 1
            if actual['metadata'].get('job_to_be_done') == expected['metadata'].get('job_to_be_done'):
                score += 1
            if len(actual['metadata'].get('input_documents', [])) == len(expected['metadata'].get('input_documents', [])):
                score += 1
        
        # Check extracted sections
        if 'extracted_sections' in actual and 'extracted_sections' in expected:
            total_checks += 2
            actual_sections = len(actual['extracted_sections'])
            expected_sections = len(expected['extracted_sections'])
            if abs(actual_sections - expected_sections) <= 2:  # Allow some variation
                score += 1
            
            # Check if we have reasonable importance rankings
            if actual_sections > 0:
                rankings = [s.get('importance_rank', 0) for s in actual['extracted_sections']]
                if sorted(rankings) == list(range(1, len(rankings) + 1)):
                    score += 1
        
        # Check subsection analysis
        if 'subsection_analysis' in actual and 'subsection_analysis' in expected:
            total_checks += 1
            if len(actual['subsection_analysis']) > 0:
                score += 1
        
        accuracy = score / total_checks if total_checks > 0 else 0
        logging.info(f"📈 Comparison score: {score}/{total_checks} ({accuracy:.1%})")
        
        return accuracy >= 0.6  # 60% similarity threshold
        
    except Exception as e:
        logging.error(f"❌ Error comparing outputs: {e}")
        return False


if __name__ == "__main__":
    main()
