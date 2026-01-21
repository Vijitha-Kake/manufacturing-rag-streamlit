# Decision Intelligence Platform for Manufacturing Defect Analysis (RAG)

## Overview
This project implements an end-to-end Retrieval-Augmented Generation (RAG) system
to analyze manufacturing defects and root causes using unstructured documents.

## Architecture
AWS S3 (optional) → LangChain Ingestion & Chunking  
→ FAISS Vector Store → OpenAI Embeddings  
→ Streamlit UI

<img width="528" height="792" alt="image" src="https://github.com/user-attachments/assets/30732965-b95d-4cfa-a9b3-fcc644afc608" />


## Features
- PDF ingestion & chunking
- Semantic search using FAISS
- Context-aware LLM responses
- Source attribution
- Streamlit-based interactive UI

## Tech Stack
Python, LangChain, FAISS, OpenAI API, Streamlit, AWS (S3, EC2 – optional)

## 🚀 Live Demo
https://manufacturing-rag-app-6habrndrkxoos39atcuv6d.streamlit.app/


## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py




