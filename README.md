# hap_dup_det: Prototype Duplicate Detection Service
This prototype detects potential duplicate content using embeddings and similarity comparison. It is served as an API using **FastAPI**, fully **dockerized**, and uses **FAISS** as the vector store for simplicity.

## Overview

### Optimized Inference
New content embeddings and chunking are guided by a metadata-based rule system for faster processing. By leveraging metadata, the system focuses on relevant parts of the content, making it **cheaper and possibly faster than fully Semantic-based chunking** while still being **more accurate than blind, uniform chunking** that ignores content structure or category.

### Candidate Selection Strategy
Similarity scores are aggregated across candidates to identify the most likely duplicates efficiently. Aggregating scores across multiple chunks prevents single-chunk noise from dominating the similarity decision, providing more robust duplicate detection compared to selecting candidates based on a single highest score.

### Symmetric Overlap Comparison
A custom function evaluates candidate content similarity more accurately by considering overlap between content chunks. This ensures that a small portion of matching content **does not falsely label two documents as duplicates**. Even if some content overlaps, the system can recognize that the overall documents are not similar, reducing false positives compared to standard similarity metrics.

## Potential Improvements
- Make metadata-based chunking modular per use case, category, or format.  
- Benchmark chunking and embedding strategies during document registration.  
- Evaluate indexing per document type and consider guided summarization for faster retrieval.  

## Assumptions
- Using **FAISS** as the vector store (with its current limitations).  
- Content types assumed: **Q&A, blog posts**.  
- Document format assumed: **Markdown (MD)**.  

## Getting Started

### Requirements
- Python 3.10+  
- Docker (for containerized deployment)  

### Running Locally
1. Clone the repository:  
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```
2. Build and run using Docker Compose:

  ```bash
  docker compose -f docker-compose.yml up --build -d
   ```
3. Access the API at: http://localhost:8000

