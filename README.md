# Persona-Driven Document Intelligence System

**Round 1B: Connect What Matters — For the User Who Matters**

A sophisticated system that extracts and prioritizes relevant sections from documents based on a specific persona and their job-to-be-done.

## 🎯 Overview

This system functions as an "intelligent document analyst" that:
- Extracts relevant sections from a collection of documents (3-10 PDF files)
- Prioritizes these sections based on persona and job requirements
- Provides granular subsection analysis
- Outputs results in the required JSON format

## 🚀 Quick Start with Docker

### 1. Clone and Prepare
```bash
git clone <repository-url>
cd 1b
```

### 2. Prepare Input Structure
**IMPORTANT**: This the input structure for 1b:

```
input/
├── challenge1b_input.json     # Configuration file
└── PDFs/                      # PDF documents directory
    ├── document1.pdf
    ├── document2.pdf
    └── document3.pdf
```

Create a `challenge1b_input.json` file in the `input/` directory:
```json
{
  "persona": {
    "role": "PhD Researcher in Computational Biology"
  },
  "job_to_be_done": {
    "task": "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks."
  },
  "documents": [
    {
      "filename": "document1.pdf",
      "title": "Graph Neural Networks for Drug Discovery - Part 1"
    },
    {
      "filename": "document2.pdf", 
      "title": "Advanced Methodologies in Computational Biology"
    },
    {
      "filename": "document3.pdf",
      "title": "Performance Benchmarks for ML in Biology"
    }
  ]
}
```

**PDF Organization Options:**
- **✅ RECOMMENDED**: Place PDFs in `input/PDFs/` directory (most reliable)
- **⚠️ FALLBACK**: Place PDFs directly in `input/` directory (system will auto-detect)

The system includes comprehensive fallback detection and will search multiple locations automatically.

### 3. Build Docker Image
```bash
docker build --platform linux/amd64 -t persona-doc-intel:1b .
```

### 4. Run the System
```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none persona-doc-intel:1b
```

### 5. Check Results
The output will be saved to `output/challenge1b_output.json`

## 📋 Input Format Requirements

### Configuration File Structure

Your input configuration must follow this **exact JSON format**:

```json
{
  "persona": {
    "role": "Travel Planner"
  },
  "job_to_be_done": {
    "task": "Plan a comprehensive 2-week itinerary for Japan covering cultural sites, cuisine, and transportation."
  },
  "documents": [
    {
      "filename": "japan_travel_guide.pdf",
      "title": "Complete Japan Travel Guide 2024"
    },
    {
      "filename": "tokyo_attractions.pdf",
      "title": "Top 50 Tokyo Attractions and Activities"
    },
    {
      "filename": "japan_transportation.pdf",
      "title": "Japan Transportation System Guide"
    }
  ]
}
```

### Directory Structure Options

**🎯 OPTION 1 - RECOMMENDED (Most Reliable):**
```
input/
├── challenge1b_input.json     # Your configuration file
└── PDFs/                      # All PDF files here
    ├── japan_travel_guide.pdf
    ├── tokyo_attractions.pdf
    └── japan_transportation.pdf
```

**🔄 OPTION 2 - FALLBACK (Auto-detected):**
```
input/
├── challenge1b_input.json     # Your configuration file
├── japan_travel_guide.pdf     # PDFs directly in input/
├── tokyo_attractions.pdf
└── japan_transportation.pdf
```

### Smart PDF Detection System

Our system includes **comprehensive PDF directory fallback** with 7-priority search:

1. ✅ Same directory as input file (`input/`)
2. ✅ PDFs subdirectory (`input/PDFs/`) - **RECOMMENDED**
3. ✅ docs subdirectory (`input/docs/`)
4. ✅ documents subdirectory (`input/documents/`)
5. ✅ Any subdirectory containing PDF files
6. ✅ Parent directories
7. ✅ Common absolute locations

**✨ System will automatically find your PDFs regardless of structure!**

### Expected Execution Commands

For evaluation, use these exact commands:

```bash
# Build the Docker image
docker build --platform linux/amd64 -t persona-doc-intel:1b .

# Run the system
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none persona-doc-intel:1b
```

## 🏗️ System Architecture

### Core Components

1. **DocumentProcessor** - Extracts text and structure from PDF documents
2. **PersonaAnalyzer** - Analyzes persona and job requirements
3. **SectionExtractor** - Identifies and extracts relevant sections
4. **RelevanceRanker** - Prioritizes sections by relevance
5. **OutputFormatter** - Generates required JSON output

### Key Features

- **Multi-factor Relevance Scoring**: Combines keyword matching, section type relevance, content quality, and job specificity
- **Persona-Specific Analysis**: Tailored processing for different user types (researcher, student, analyst, etc.)
- **Intelligent Section Detection**: Identifies sections based on formatting, patterns, and content structure
- **Subsection Analysis**: Provides granular content breakdown
- **Diversity Balancing**: Ensures coverage across multiple documents and section types

## 📊 Output Format

The system generates output in the required JSON format:

```json
{
  "metadata": {
    "input_documents": ["doc1.pdf", "doc2.pdf"],
    "persona": "PhD Researcher in Computational Biology",
    "job_to_be_done": "Prepare literature review...",
    "processing_timestamp": "2024-01-01T12:00:00"
  },
  "extracted_sections": [
    {
      "document": "doc1.pdf",
      "page_number": 5,
      "section_title": "Methodology",
      "importance_rank": 0.95
    }
  ],
  "subsection_analysis": [
    {
      "document": "doc1.pdf",
      "subsection_id": "Methodology_1",
      "refined_text": "The experimental setup...",
      "page_number": 5
    }
  ]
}
```

## 🎯 Supported Personas

The system is optimized for various personas:

- **Researchers**: Focus on methodology, findings, analysis, conclusions
- **Students**: Emphasis on concepts, theories, examples, summaries
- **Analysts**: Priority on analysis, trends, performance, recommendations
- **Investors**: Focus on financial data, revenue, growth, market analysis
- **Entrepreneurs**: Emphasis on strategy, market analysis, business models
- **Journalists**: Priority on summaries, key points, highlights, trends

## ⚡ Performance Constraints

- **CPU Only**: No GPU requirements
- **Model Size**: ≤ 1GB total
- **Processing Time**: ≤ 60 seconds for 3-5 documents
- **No Internet Access**: Fully offline operation

## 🔧 Manual Installation

If you prefer to run without Docker:

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .

# Run the system
python main.py
```

## 📁 Project Structure

```
1b/
├── main.py                              # Main entry point
├── requirements.txt                     # Python dependencies
├── Dockerfile                          # Docker configuration
├── setup.py                            # Package setup
├── pyproject.toml                      # Modern packaging config
├── README.md                           # This file
├── persona_document_intelligence/      # Core package
│   ├── __init__.py
│   ├── core/                          # Core components
│   │   ├── document_processor.py
│   │   ├── persona_analyzer.py
│   │   ├── section_extractor.py
│   │   └── relevance_ranker.py
│   └── utils/                         # Utilities
│       └── output_formatter.py
├── input/                             # Input directory
│   ├── challenge1b_input.json        # Configuration file (REQUIRED)
│   └── PDFs/                         # PDF documents (RECOMMENDED)
│       ├── document1.pdf
│       ├── document2.pdf
│       └── document3.pdf
└── output/                            # Output directory
    └── challenge1b_output.json        # Generated output
```

## 🧪 Testing

### Test Cases

The system is designed to handle the provided test cases:

1. **Academic Research**: 4 research papers on "Graph Neural Networks for Drug Discovery"
2. **Business Analysis**: 3 annual reports from competing tech companies
3. **Educational Content**: 5 chapters from organic chemistry textbooks

### Sample Configuration Examples

**Example 1 - Academic Research:**
```json
{
  "persona": {
    "role": "PhD Researcher in Computational Biology"
  },
  "job_to_be_done": {
    "task": "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks."
  },
  "documents": [
    {
      "filename": "paper1.pdf",
      "title": "Graph Neural Networks for Drug Discovery"
    },
    {
      "filename": "paper2.pdf",
      "title": "Deep Learning in Molecular Biology"
    },
    {
      "filename": "paper3.pdf",
      "title": "Computational Methods for Drug Development"
    }
  ]
}
```

**Example 2 - Business Analysis:**
```json
{
  "persona": {
    "role": "Business Analyst"
  },
  "job_to_be_done": {
    "task": "Analyze market trends and competitive positioning for quarterly business review."
  },
  "documents": [
    {
      "filename": "q3_report.pdf",
      "title": "Q3 2024 Financial Report"
    },
    {
      "filename": "market_analysis.pdf",
      "title": "Industry Market Analysis"
    },
    {
      "filename": "competitor_review.pdf",
      "title": "Competitive Intelligence Report"
    }
  ]
}
```

**Example 3 - Travel Planning:**
```json
{
  "persona": {
    "role": "Travel Planner"
  },
  "job_to_be_done": {
    "task": "Create detailed travel itinerary with cultural sites, dining, and logistics."
  },
  "documents": [
    {
      "filename": "destination_guide.pdf",
      "title": "Complete Destination Travel Guide"
    },
    {
      "filename": "local_cuisine.pdf",
      "title": "Local Food and Restaurant Guide"
    },
    {
      "filename": "transportation.pdf",
      "title": "Transportation and Logistics Guide"
    }
  ]
}
```

## 🛠️ Troubleshooting

### Common Issues

1. **Docker build fails**: Ensure you're using the correct platform flag
2. **No PDF files found**: Check that PDF files are in the input directory
3. **Memory issues**: The system is optimized for CPU-only operation
4. **Slow processing**: Processing time is typically 30-60 seconds for 3-5 documents

### Performance Optimization

- The system uses efficient sentence transformers (all-MiniLM-L6-v2)
- Implements early termination for large documents
- Uses optimized text processing algorithms
- Leverages multi-threading where appropriate

## 📈 Scoring Criteria

The system is optimized for the evaluation criteria:

- **Section Relevance (60 points)**: How well selected sections match persona + job requirements
- **Sub-Section Relevance (40 points)**: Quality of granular subsection extraction and ranking

## 🤝 Contributing

This system is designed for the Adobe Hack Round 1B challenge. The architecture is modular and extensible for future enhancements.

## 📄 License

MIT License - see LICENSE file for details. 