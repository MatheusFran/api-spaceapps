# 🚀 SpaceLifeTeam - AI-Powered Space Research Discovery

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![LangChain](https://img.shields.io/badge/langchain-0.1+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

An intelligent search system for space research articles, powered by FAISS vector embeddings and semantic search technology. SpaceLifeTeam helps researchers, students, and space enthusiasts discover relevant scientific literature efficiently.

## ✨ Features

- **Semantic Search**: Uses FAISS vector embeddings for intelligent document retrieval
- **Smart Deduplication**: Automatically removes duplicate articles from search results
- **Relevant Excerpts**: Extracts and displays the most relevant passages from each document
- **User-Friendly Interface**: Clean, intuitive Streamlit-based UI
- **Fast & Efficient**: Cached embeddings for quick response times
- **Batch Processing**: Automated pipeline for processing large datasets of research articles

## 🏗️ Architecture

The project consists of two main components:

### 1. Data Pipeline (`pipeline_transform.py`)
- Extracts articles from CSV containing publication links
- Downloads and caches documents from web sources
- Splits documents into manageable chunks
- Generates embeddings using HuggingFace models
- Creates and saves FAISS vector store

### 2. Search Interface (`streamlit.py`)
- Streamlit web application
- Real-time semantic search
- Smart result ranking and deduplication
- Interactive document exploration

### 🎯 Next Steps
We're continuously improving SpaceLifeTeam! 

Here's what's coming next:
🔮 Planned Features

**Expanded Data Sources**:
Integration with additional scientific databases (arXiv, PubMed, NASA Technical Reports)
Support for PDF parsing and processing
Real-time data updates and synchronization

**LLM Integration for Intelligent Reports**:
Automatic research summaries generation
Comparative analysis across multiple papers
Citation network visualization
Key findings extraction and synthesis
Research gap identification
Personalized research recommendations
