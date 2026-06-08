from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from agents.planner_agent import generate_plan
from agents.pubmed_agent import search_pubmed
from agents.analysis_agent import analyze_papers
from agents.report_agent import generate_report
from agents.evidence_agent import classify_study_types
from retrieval.retriever import DocumentRetriever

class GraphState(TypedDict):
    query: str
    plan: str
    raw_papers: List[dict]
    retrieved_papers: List[dict]
    analysis: str
    report: str
    task_id: str

def safe_update_status(state: GraphState, status: str, message: str):
    task_id = state.get("task_id")
    if task_id:
        try:
            from backend.database import update_task_status
            update_task_status(task_id, status, message)
        except Exception as e:
            print(f"Failed to update task status in DB: {e}")
    else:
        print(f"[{status}] {message}")

def plan_node(state: GraphState):
    query = state["query"]
    safe_update_status(state, "PLANNING", "Analyzing research query and structuring plan...")
    plan = generate_plan(query)
    return {"plan": plan}

def fetch_pubmed_node(state: GraphState):
    query = state["query"]
    safe_update_status(state, "FETCHING_PUBMED", "Searching and fetching abstracts from PubMed...")
    # We fetch a larger pool of papers from PubMed to index
    papers = search_pubmed(query, max_results=10)
    return {"raw_papers": papers}

def evidence_node(state: GraphState):
    papers = state.get("raw_papers", [])
    safe_update_status(state, "CLASSIFYING_EVIDENCE", f"Classifying clinical evidence levels for {len(papers)} papers...")
    if papers:
        papers = classify_study_types(papers)
    return {"raw_papers": papers}

def retrieve_node(state: GraphState):
    query = state["query"]
    papers = state.get("raw_papers", [])
    safe_update_status(state, "RETRIEVING", "Indexing papers into FAISS and running similarity search...")
    
    # Initialize retriever
    retriever = DocumentRetriever()
    
    if papers:
        # Index papers into FAISS
        retriever.add_papers_to_index(papers)
        # Retrieve the top most relevant ones based on embeddings
        top_papers = retriever.get_relevant_papers(query, top_k=3)
    else:
        top_papers = []
        
    return {"retrieved_papers": top_papers}

def analyze_node(state: GraphState):
    query = state["query"]
    papers = state.get("retrieved_papers", [])
    safe_update_status(state, "ANALYZING", f"Distilling clinical findings from {len(papers)} retrieved papers...")
    if not papers:
        return {"analysis": "No relevant papers found to analyze."}
        
    analysis = analyze_papers(query, papers)
    return {"analysis": analysis}

def report_node(state: GraphState):
    query = state["query"]
    analysis = state.get("analysis", "")
    papers = state.get("retrieved_papers", [])
    safe_update_status(state, "COMPILING_REPORT", "Assembling final research summary and compiling sources...")
    report = generate_report(query, analysis, papers)
    return {"report": report}

def build_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("planner", plan_node)
    workflow.add_node("pubmed_fetcher", fetch_pubmed_node)
    workflow.add_node("evidence_classifier", evidence_node)
    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("analyzer", analyze_node)
    workflow.add_node("reporter", report_node)
    
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "pubmed_fetcher")
    workflow.add_edge("pubmed_fetcher", "evidence_classifier")
    workflow.add_edge("evidence_classifier", "retriever")
    workflow.add_edge("retriever", "analyzer")
    workflow.add_edge("analyzer", "reporter")
    workflow.add_edge("reporter", END)
    
    return workflow.compile()

if __name__ == "__main__":
    app = build_graph()
    query = "Treatments for Alzheimer's Disease"
    print(f"Running pipeline for query: {query}")
    final_state = app.invoke({"query": query, "task_id": ""})
    print("\n--- FINAL REPORT ---\n")
    print(final_state["report"])

