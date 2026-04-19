# ProKnow-RAG Architecture

## System Overview

ProKnow-RAG is a modular pure-retrieval knowledge system for technical documents, academic papers, and GitHub projects.

## Architecture

- **Indexing Pipeline**: Data Preparation → Index Construction → Qdrant Storage
- **Query Pipeline**: User Query → Retrieval Optimization → Return Results
- **Cross-cutting**: Config, Logging, Cache, Evaluation

## Key Design Decisions

- Single Qdrant Collection with dense(1024) + sparse + ColBERT(128) vectors
- BGE-M3 unified embedding model
- BGE-reranker-v2-m3 Chinese Cross-Encoder reranking
- Differentiated chunking and retrieval weights per data type
