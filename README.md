# Agentic AI Medical Research Assistant 🧬

A multi-agent AI system that helps users explore biomedical research papers and generates evidence-based summaries using open-source tools, Hugging Face API, Google Gemini, and the PubMed/PMC databases.

## 🏗️ System Architecture & Multi-Agent System
- **Planner Agent**: Analyzes the query and structures the task plan.
- **PubMed & PMC Fetcher**: Searches and fetches biomedical abstracts, and automatically downloads open-access **full-text articles** from PubMed Central (PMC) when available.
- **RAG & Embeddings**: Creates a local FAISS index using semantic embeddings for vector similarity search.
- **Retriever Agent**: Similarity search for top-k relevant paragraphs against the vector DB.
- **Analysis Agent**: Connects to Google GenAI (`gemini-2.5-flash`) to extract key findings, treatment details, and **disease symptoms**.
- **Report Agent**: Compiles the summarized research with cited sources.
- **Medical Safety Verifier Agent**: Safety guardrail that cross-references report claims against source texts to flag any medical hallucinations, appending a **Medical Verification Warning/Safety Seal**.
- **Evaluation Module**: Evaluates the RAG pipeline using local embedding-based cosine similarity and lexical (Jaccard similarity) overlap metrics.
- **Diagnostic Analytics**: Integrated diagnostic metrics directly in the history view tracking system performance and job stats.

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
   docker run -p 8501:8501 -e HUGGINGFACEHUB_API_TOKEN=your_hf_token_here -e GEMINI_API_KEY=your_gemini_key_here medical-assistant
   ```
   Then visit `http://localhost:8501` in your browser.
