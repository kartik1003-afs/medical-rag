# Agentic AI Medical Research Assistant 🧬

A multi-agent AI system that helps users explore biomedical research papers and generates evidence-based summaries using ONLY open-source tools, free HF inference APIs, and the PubMed free database.

## 🏗️ System Architecture & Multi-Agent System
- **Planner Agent**: Analyzes the query and structures the task plan.
- **PubMed Search Agent**: Searches and fetches biomedical abstracts via the National Library of Medicine E-Utilities API.
- **RAG & Embeddings**: Creates a local FAISS index using `sentence-transformers/all-MiniLM-L6-v2` embeddings downloaded from HF Hub.
- **Retriever Agent**: Similarity search for top-k contextual abstracts against the vector DB.
- **Analysis Agent**: Connects to Hugging Face Serverless Inference API (`Mistral-7B-Instruct`) to distill key findings.
- **Report Agent**: Compiles the summarized research with cited sources.
- **Evaluation Module**: Implements RAGAS framework (faithfulness, answer relevance, context precision, context recall) to objectively output metrics using HF evaluator models.

## 🛠️ Tech Stack
- **Dependency Management**: `uv`
- **Orchestration**: `langgraph`
- **Retrieval Engine**: `faiss-cpu`, `langchain-huggingface`
- **Frontend**: Streamlit
- **Containerization**: Docker

## 🚀 How to Run Locally with `uv`

1. **Prerequisites**: Make sure `uv` is installed (`pip install uv` or via standalone script).
2. **Install Dependencies**:
   ```bash
   uv sync
   ```
3. **Environment Variables**:
   It requires a Hugging Face API key to run embeddings and a Gemini API key for inferences.
   ```bash
   # Windows PowerShell
   $env:HUGGINGFACEHUB_API_TOKEN="your_hf_access_token_here"
   $env:GEMINI_API_KEY="your_gemini_key_here"
   
   # Linux/Mac
   export HUGGINGFACEHUB_API_TOKEN="your_hf_access_token_here"
   export GEMINI_API_KEY="your_gemini_key_here"
   ```
   *Note: You can also just enter these directly in the Streamlit Sidebar UI!*
   
4. **Launch Application**:
   ```bash
   uv run streamlit run frontend/app.py
   ```

## 🐳 How to Run using Docker

1. **Build the image**:
   ```bash
   docker build -t medical-assistant .
   ```
2. **Run the container**:
   ```bash
   docker run -p 8501:8501 -e HF_TOKEN=your_token_here medical-assistant
   ```
   Then visit `http://localhost:8501` in your browser.
