import streamlit as st
import os
import sys
import pandas as pd
import time
import requests
import plotly.express as px

# Add root directory to python path for internal imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(page_title="Medical Research Intelligence Dashboard", page_icon="🧬", layout="wide")

# Backend URL configuration
BACKEND_URL = "http://localhost:8000"

st.title("Medical Research Intelligence Dashboard 🧬")

# Sidebar Configuration
st.sidebar.header("Backend Configuration")
hf_token = st.sidebar.text_input("Hugging Face API Token:", type="password", help="Required for BAAI Embeddings. Leave empty if set on server environment.")
gemini_key = st.sidebar.text_input("Gemini API Key:", type="password", help="Required for Gemini LLMs. Leave empty if set on server environment.")

# Helper function to render research results (reused for new runs and history)
def render_research_results(results):
    # 1. Retrieved Papers Table
    st.subheader("Retrieved Papers (Sources)")
    retrieved_papers = results.get("retrieved_papers", [])
    if retrieved_papers:
        df_retrieved = pd.DataFrame(retrieved_papers)
        display_cols = ['title', 'authors', 'journal', 'year']
        st.dataframe(df_retrieved[display_cols] if all(c in df_retrieved for c in display_cols) else df_retrieved)
    else:
        st.info("No papers retrieved.")
        return
        
    # 2. Research Summary
    st.subheader("Generated Research Summary")
    st.markdown(results.get("report", "No report generated."))
    
    # 3. Local Evaluation Metrics Section
    st.divider()
    st.header("Evaluation Metrics (Local Semantic Engine)")
    metrics = results.get("metrics", {})
    if metrics and "error" not in metrics:
        # Create columns for metric cards
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Faithfulness Score", metrics.get("faithfulness", 0.0))
        mc2.metric("Answer Relevancy Score", metrics.get("answer_relevancy", 0.0))
        mc3.metric("Context Precision", metrics.get("context_precision", 0.0))
        mc4.metric("Context Recall", metrics.get("context_recall", 0.0))
        
        # Plotly chart
        df_metrics = pd.DataFrame([
            {"Metric": "Faithfulness", "Score": metrics.get("faithfulness", 0.0)},
            {"Metric": "Answer Relevance", "Score": metrics.get("answer_relevancy", 0.0)},
            {"Metric": "Context Precision", "Score": metrics.get("context_precision", 0.0)},
            {"Metric": "Context Recall", "Score": metrics.get("context_recall", 0.0)},
        ])
        fig_metrics = px.bar(
            df_metrics, 
            x="Metric", 
            y="Score", 
            range_y=[0, 1], 
            title="Local RAG Semantic & Overlap Evaluation",
            color="Score",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_metrics, use_container_width=True)
        if "warning" in metrics:
            st.warning(f"Note: {metrics['warning']}")
    elif metrics and "error" in metrics:
        st.warning(f"Evaluation metrics unavailable: {metrics.get('error')}")
    else:
        st.info("No evaluation metrics generated.")

# Tabs Setup
tab1, tab2, tab3 = st.tabs(["🧬 New Research", "📜 Research History", "📊 Dashboard Stats"])

# Tab 1: New Research Run
with tab1:
    st.header("Start a New Research Pipeline")
    query = st.text_input("Enter your research topic:", placeholder="e.g., Treatments for Alzheimer's Disease", key="query_input")
    
    if st.button("Run Research Pipeline", key="run_btn"):
        if not query.strip():
            st.warning("Please enter a research query.")
        else:
            # Prepare request payload
            payload = {
                "query": query,
                "hf_token": hf_token if hf_token.strip() else None,
                "gemini_key": gemini_key if gemini_key.strip() else None
            }
            
            with st.spinner("Submitting research job to backend..."):
                try:
                    response = requests.post(f"{BACKEND_URL}/api/research/submit", json=payload)
                    response.raise_for_status()
                    job_data = response.json()
                    task_id = job_data["task_id"]
                    st.success(f"Job submitted! Task ID: {task_id}")
                except Exception as e:
                    st.error(f"Failed to submit job to backend: {str(e)}")
                    st.stop()
            
            # Polling placeholder
            status_container = st.empty()
            progress_bar = st.progress(0)
            
            # Map status code to rough percentage
            status_map = {
                "PENDING": (10, "Job queued..."),
                "PLANNING": (25, "📝 Planning research steps..."),
                "FETCHING_PUBMED": (45, "🔍 Fetching abstracts from PubMed..."),
                "CLASSIFYING_EVIDENCE": (65, "⚖️ Classifying clinical study designs..."),
                "RETRIEVING": (80, "🗄️ Indexing and retrieving via FAISS..."),
                "ANALYZING": (90, "🧠 Distilling clinical findings..."),
                "COMPILING_REPORT": (95, "📋 Assembling final report..."),
                "EVALUATING": (98, "📊 Running semantic evaluation..."),
                "COMPLETED": (100, "✅ Completed!"),
                "FAILED": (100, "❌ Failed.")
            }
            
            while True:
                try:
                    status_response = requests.get(f"{BACKEND_URL}/api/research/status/{task_id}")
                    status_response.raise_for_status()
                    task_data = status_response.json()
                    
                    status = task_data.get("status", "PENDING")
                    progress_msg = task_data.get("progress_msg", "")
                    
                    pct, msg = status_map.get(status, (10, "Processing..."))
                    
                    with status_container.container():
                        st.info(f"**Task Status:** {status} | **Details:** {progress_msg}")
                        st.write(f"Current Phase Log: *{msg}*")
                    
                    progress_bar.progress(pct)
                    
                    if status == "COMPLETED":
                        st.success("Pipeline Execution Complete!")
                        time.sleep(1)
                        status_container.empty()
                        progress_bar.empty()
                        render_research_results(task_data)
                        break
                    elif status == "FAILED":
                        st.error(f"Pipeline failed: {task_data.get('error_message', 'Unknown error')}")
                        break
                        
                except Exception as e:
                    st.error(f"Error checking status: {str(e)}")
                    break
                    
                time.sleep(2.5)

# Tab 2: Research History Viewer
with tab2:
    st.header("Search & Retrieve Historical Runs")
    
    # Refresh button
    if st.button("Refresh History"):
        st.rerun()
        
    try:
        history_response = requests.get(f"{BACKEND_URL}/api/research/history")
        history_response.raise_for_status()
        history = history_response.json()
    except Exception as e:
        st.error(f"Failed to fetch research history from backend: {str(e)}")
        history = []
        
    if not history:
        st.info("No historical research runs found. Run a new pipeline to save queries!")
    else:
        # Create a clean selection view
        df_history = pd.DataFrame(history)
        df_history['display_name'] = df_history.apply(
            lambda r: f"[{r['created_at'][:16].replace('T', ' ')}] {r['query']} ({r['status']})", 
            axis=1
        )
        
        selected_display = st.selectbox(
            "Select a previous research run:",
            options=df_history['display_name'].tolist()
        )
        
        if selected_display:
            # Extract row matching select box
            selected_row = df_history[df_history['display_name'] == selected_display].iloc[0]
            task_id = selected_row['id']
            
            # Fetch detailed task info
            with st.spinner("Fetching historical report details..."):
                try:
                    task_res = requests.get(f"{BACKEND_URL}/api/research/status/{task_id}")
                    task_res.raise_for_status()
                    task_detail = task_res.json()
                    
                    # Notes and Delete Options layout
                    h_col1, h_col2 = st.columns([2, 1])
                    with h_col1:
                        st.write(f"**Task ID:** `{task_detail['id']}`")
                        st.write(f"**Run Date:** `{task_detail['created_at']}`")
                    
                    with h_col2:
                        # Delete Task button
                        if st.button("🗑️ Delete this Research Run", key=f"delete_{task_id}"):
                            try:
                                del_res = requests.delete(f"{BACKEND_URL}/api/research/delete/{task_id}")
                                del_res.raise_for_status()
                                st.success("Research run deleted successfully!")
                                time.sleep(1)
                                st.rerun()
                            except Exception as del_err:
                                st.error(f"Failed to delete run: {str(del_err)}")
                                
                    # Custom Notes section
                    existing_notes = task_detail.get("notes", "") or ""
                    updated_notes = st.text_area("Research Notes & Annotation:", value=existing_notes, height=120, key=f"notes_{task_id}")
                    if st.button("Save Notes", key=f"save_notes_{task_id}"):
                        try:
                            notes_res = requests.post(f"{BACKEND_URL}/api/research/notes/{task_id}", json={"notes": updated_notes})
                            notes_res.raise_for_status()
                            st.success("Notes saved successfully!")
                        except Exception as note_err:
                            st.error(f"Failed to save notes: {str(note_err)}")
                            
                    st.divider()
                    
                    if task_detail['status'] == 'COMPLETED':
                        render_research_results(task_detail)
                    elif task_detail['status'] == 'FAILED':
                        st.error(f"This run failed with error: {task_detail.get('error_message')}")
                    else:
                        st.warning(f"This run is in state '{task_detail['status']}'. Wait for it to finish or check progress in New Research.")
                except Exception as e:
                    st.error(f"Failed to load details for task {task_id}: {str(e)}")

# Tab 3: Dashboard Stats
with tab3:
    st.header("System Diagnostic Dashboard & Analytics")
    if st.button("Refresh Dashboard Stats", key="refresh_stats_btn"):
        st.rerun()
        
    try:
        stats_response = requests.get(f"{BACKEND_URL}/api/research/stats")
        stats_response.raise_for_status()
        stats = stats_response.json()
        
        # Diagnostic Metric Cards
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Total Research Jobs", stats.get("total_jobs", 0))
        col_s2.metric("Successful Jobs", stats.get("completed_jobs", 0))
        col_s3.metric("Failed Jobs", stats.get("failed_jobs", 0))
        col_s4.metric("Success Rate", f"{stats.get('success_rate', 0.0)}%")
        
        st.divider()
        st.subheader("Top Research Queries")
        
        pop_queries = stats.get("popular_queries", [])
        if pop_queries:
            df_pop = pd.DataFrame(pop_queries)
            # Create a nice horizontal bar chart
            fig_pop = px.bar(
                df_pop, 
                x="count", 
                y="query", 
                orientation='h', 
                labels={"count": "Number of Runs", "query": "Research Topic"},
                color="count",
                color_continuous_scale="Plasma",
                title="Top 5 Research Topics Run"
            )
            st.plotly_chart(fig_pop, use_container_width=True)
        else:
            st.info("No query logs recorded in database yet.")
    except Exception as e:
        st.error(f"Failed to load dashboard metrics from backend: {str(e)}")
