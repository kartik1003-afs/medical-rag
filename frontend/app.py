import streamlit as st
import os
import sys
import pandas as pd
import collections
import re
import plotly.express as px
import plotly.graph_objects as go

# Fix for asyncio 'Event loop is closed' error on Windows due to Ragas/aiohttp
if sys.platform == 'win32':
    import asyncio
    import nest_asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    nest_asyncio.apply()
    
    # Silence specific WinError 10038 socket teardown issue during evaluation
    try:
        from asyncio.selector_events import _SelectorSocketTransport
        _original_write = _SelectorSocketTransport.write
        def _monkeypatch_write(self, data):
            try:
                _original_write(self, data)
            except OSError as e:
                if e.winerror == 10038:
                    pass
                else:
                    raise
        _SelectorSocketTransport.write = _monkeypatch_write
    except ImportError:
        pass

# Add root directory to python path for internal imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import build_graph
from evaluation.ragas_eval import evaluate_pipeline

st.set_page_config(page_title="Medical Research Intelligence Dashboard", page_icon="🧬", layout="wide")

st.title("Medical Research Intelligence Dashboard 🧬")

st.sidebar.header("Configuration")
hf_token = st.sidebar.text_input("Hugging Face API Token:", type="password", help="Required for BAAI Embeddings")
if hf_token:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

gemini_key = st.sidebar.text_input("Gemini API Key:", type="password", help="Required for Gemini 2.5 Flash LLM")
if gemini_key:
    os.environ["GEMINI_API_KEY"] = gemini_key

# 1. Query Input
query = st.text_input("Enter your research topic:", placeholder="e.g., Treatments for Alzheimer's Disease")

if st.button("Run Research Pipeline"):
    if not os.environ.get("HUGGINGFACEHUB_API_TOKEN") or not os.environ.get("GEMINI_API_KEY"):
        st.warning("Please provide both Hugging Face API Token and Gemini API Key in the sidebar.")
    elif not query:
        st.warning("Please enter a research query.")
    else:
        with st.spinner("Initializing LangGraph workflow..."):
            app = build_graph()
            
        with st.spinner("Running Multi-Agent System (Planner -> PubMed -> Evidence Classifier -> FAISS -> Analyzer)..."):
            try:
                state = app.invoke({"query": query})
            except Exception as e:
                st.error(f"Pipeline failed: {str(e)}")
                st.stop()
            
        st.success("Pipeline Execution Complete!")
        
        # 2. Retrieved Papers Table
        st.subheader("Retrieved Papers (Sources)")
        retrieved_papers = state.get("retrieved_papers", [])
        if retrieved_papers:
            df_retrieved = pd.DataFrame(retrieved_papers)
            display_cols = ['title', 'authors', 'journal', 'year']
            st.dataframe(df_retrieved[display_cols] if all(c in df_retrieved for c in display_cols) else df_retrieved)
        else:
            st.info("No papers retrieved.")
            st.stop()
            
        # 3. Research Summary
        st.subheader("Generated Research Summary")
        st.markdown(state.get("report", "No report generated."))
        
        # 4. Research Analytics Section
        st.divider()
        st.header("Research Analytics")
        raw_papers = state.get("raw_papers", [])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 1. Publication Trend Analysis
            st.subheader("Publication Trends")
            years = [int(p.get("year")) for p in raw_papers if p.get("year") and str(p.get("year")).isdigit()]
            if years:
                year_counts = collections.Counter(years)
                df_years = pd.DataFrame(list(year_counts.items()), columns=["Year", "Count"]).sort_values("Year")
                fig_trend = px.line(df_years, x="Year", y="Count", markers=True, title="Papers Published per Year")
                st.plotly_chart(fig_trend, width="stretch")
            else:
                st.info("No publication year data.")
                
            # 3. Keyword Frequency Form Abstracts
            st.subheader("Keyword Analysis")
            stop_words = set(["the", "and", "of", "to", "in", "a", "is", "for", "with", "that", "this", "by", "on", "as", "are", "be", "an", "from", "was", "were", "or", "it", "at", "not", "which", "study", "patients", "disease", "treatment", "may", "has", "we", "can", "these", "results", "have", "been", "their", "will", "more", "also", "such", "than", "between", "both", "all", "but"])
            all_text = " ".join([p.get("abstract", "") for p in raw_papers]).lower()
            words = re.findall(r'\b[a-z]{4,}\b', all_text)
            keywords = [w for w in words if w not in stop_words]
            if keywords:
                kw_counts = collections.Counter(keywords).most_common(10)
                df_kw = pd.DataFrame(kw_counts, columns=["Keyword", "Frequency"]).sort_values("Frequency", ascending=True)
                fig_kw = px.bar(df_kw, x="Frequency", y="Keyword", orientation='h', title="Top 10 Keywords in Abstracts")
                st.plotly_chart(fig_kw, width="stretch")
            else:
                st.info("No keywords extracted.")
                
            # 5. Paper Relevance Scores
            st.subheader("Paper Relevance Scores")
            df_relevance = pd.DataFrame(retrieved_papers)
            if 'similarity_score' in df_relevance.columns:
                df_relevance['Short Title'] = df_relevance['title'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x)
                df_relevance = df_relevance.sort_values('similarity_score', ascending=True)
                fig_rel = px.bar(df_relevance, x="similarity_score", y="Short Title", orientation='h', 
                                title="Similarity Distance of Retrieved Documents (Lower is closer)")
                st.plotly_chart(fig_rel, width="stretch")
            
        with col2:
            # 2. Top Journals Analysis
            st.subheader("Top Journals")
            journals = [p.get("journal", "Unknown") for p in raw_papers if p.get("journal")]
            if journals:
                journal_counts = collections.Counter(journals).most_common(10)
                df_journals = pd.DataFrame(journal_counts, columns=["Journal", "Count"]).sort_values("Count", ascending=True)
                fig_journal = px.bar(df_journals, x="Count", y="Journal", orientation='h', title="Most Publishing Journals")
                st.plotly_chart(fig_journal, width="stretch")
            else:
                st.info("No journal data.")
                
            # 4. Study Type Distribution
            st.subheader("Study Type Distribution")
            study_types = [p.get("study_type", "Other") for p in raw_papers]
            if study_types:
                st_counts = collections.Counter(study_types)
                df_st = pd.DataFrame(list(st_counts.items()), columns=["Study Type", "Count"])
                fig_st = px.pie(df_st, values='Count', names='Study Type', title="Distribution by Evidence Ranking Agent")
                st.plotly_chart(fig_st, width="stretch")
                
        # Drug-Disease Knowledge Graph
        st.divider()
        st.header("Drug–Disease Relationship Graph")
        with st.spinner("Extracting Knowledge Graph Entities (This may take a minute with Gemini)..."):
            # Import dynamically to ensure fast initial page loads
            from drug_disease_graph.relationship_extraction import extract_relationships
            from drug_disease_graph.graph_builder import build_graph
            from drug_disease_graph.graph_visualizer import visualize_graph_plotly
            
            all_relationships = []
            for p in retrieved_papers:
                abs_text = p.get("abstract", "")
                if abs_text.strip():
                    rels = extract_relationships(abs_text)
                    all_relationships.extend(rels)
                
            if all_relationships:
                col_r1, col_r2 = st.columns([1, 2])
                with col_r1:
                    st.subheader("Extracted Relationships")
                    for rel in all_relationships:
                        st.markdown(f"- **{rel.get('disease', 'Unknown')}** → *treated_by* → **{rel.get('drug', 'Unknown')}**")
                        
                with col_r2:
                    st.subheader("Interactive Knowledge Graph")
                    G = build_graph(all_relationships)
                    fig_graph = visualize_graph_plotly(G)
                    if fig_graph:
                        st.plotly_chart(fig_graph, width="stretch")
            else:
                st.info("No explicit drug-disease treatment relationships found in the retrieved abstracts.")

        # 5. Evaluation Metrics Section
        st.divider()
        st.header("Evaluation Metrics")
        with st.spinner("Running RAGAS Evaluator..."):
            contexts = [p.get("abstract", "") for p in retrieved_papers]
            answer = state.get("analysis", "")
            ground_truth = "Dummy truth for evaluation since query is open ended."
            
            metrics = evaluate_pipeline(query, answer, contexts, ground_truth)
            
            if "error" in metrics:
                st.error(metrics["error"])
            else:
                st.subheader("RAGAS Score Charts")
                df_metrics = pd.DataFrame([
                    {"Metric": "Faithfulness", "Score": metrics.get("faithfulness", 0)},
                    {"Metric": "Answer Relevance", "Score": metrics.get("answer_relevancy", 0)},
                    {"Metric": "Context Precision", "Score": metrics.get("context_precision", 0)},
                    {"Metric": "Context Recall", "Score": metrics.get("context_recall", 0)},
                ])
                fig_metrics = px.bar(df_metrics, x="Metric", y="Score", range_y=[0, 1], title="RAGAS Evaluation Scores")
                st.plotly_chart(fig_metrics, width="stretch")
