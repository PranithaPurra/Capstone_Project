# Zepto Support Assistant

## Overview

This module implements an offline-first support assistant for Zepto policy questions.

It uses local embeddings, ChromaDB retrieval, LangGraph orchestration, Pydantic validation and FastAPI.

## Architecture

```text
Zepto Policy Documents
        ↓
Sentence Transformer
        ↓
ChromaDB
        ↓
User Query
        ↓
LangGraph
        ↓
classify_intent
        ↓
retrieve_and_answer / direct_answer
        ↓
Pydantic Response
        ↓
FastAPI
```

## Policy Corpus

The module contains eight policy documents:

- doc_01.txt
- doc_02.txt
- doc_03.txt
- doc_04.txt
- doc_05.txt
- doc_06.txt
- doc_07.txt
- doc_08.txt

## Embeddings

Uses `sentence-transformers/all-MiniLM-L6-v2`.

Document embeddings are stored in ChromaDB.

## LangGraph

The graph contains three nodes:
- classify_intent
- retrieve_and_answer
- direct_answer

Conditional routing sends policy questions to retrieval and other questions to the direct-answer path.

## Mock Mode

The application uses deterministic mock mode by default.

`MOCK_LLM=1`

## API

### POST /ask

Request:

```json
{"query": "How much does delivery cost?"}
```

Response fields:
- answer
- sources
- confidence

### GET /health

Returns the application health status.

## Docker

Build:

```bash
docker build -t zepto-support-assistant .
```

Run:

```bash
docker run -p 7860:7860 zepto-support-assistant
```

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```