FROM --platform=linux/amd64 python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables for better performance
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    VECLIB_MAXIMUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the package files
COPY persona_document_intelligence/ /app/persona_document_intelligence/
COPY main.py pyproject.toml /app/

# Install the package in development mode
RUN pip install -e .

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Set the entrypoint
ENTRYPOINT ["python", "main.py"] 