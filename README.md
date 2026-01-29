# Local Codebase Analysis Agent

## Overview

This project is a **local code analysis assistant** built with a local LLM using **Ollama** and **LangChain**.

It allows you to:
- Automatically scan a software project
- Index source code files across multiple languages
- Generate embeddings and store them in a local vector database
- Ask questions **strictly based on the project’s code**
- Clearly state when information is not present in the codebase
- Always reference the file when quoting code

The assistant runs in the terminal and can be queried using natural language.

---

## How It Works

1. The current project directory is scanned recursively
2. Source files are filtered by extension
3. Files are split into chunks
4. Chunks are embedded using an Ollama embedding model
5. Embeddings are stored locally using Chroma
6. User questions trigger a contextual similarity search
7. The LLM generates answers **only from the retrieved code context**

---

## Requirements

### System Requirements
- **Python 3.10+**
- **Ollama** installed and running

### Required Ollama Models
```bash
ollama pull qwen2.5-coder:3b
ollama pull nomic-embed-text
