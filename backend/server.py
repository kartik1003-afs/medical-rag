import os
import sys
import uuid
import logging
from typing import Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.database import init_db, create_task, update_task_status, update_task_result, get_task, get_all_tasks, delete_task, update_task_notes, get_system_stats
from main import build_graph
from evaluation.local_eval import evaluate_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medical_research_backend")

app = FastAPI(title="Medical Research Assistant API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
def startup_event():
    init_db()
    logger.info("Database initialized.")

class ResearchRequest(BaseModel):
    query: str
    hf_token: Optional[str] = None
    gemini_key: Optional[str] = None

class NotesRequest(BaseModel):
    notes: str

def run_agent_pipeline(task_id: str, query: str, hf_token: Optional[str], gemini_key: Optional[str]):
    """Background task to run the LangGraph pipeline and evaluate it."""
    # Set environment variables for this thread/process execution if provided
    if hf_token:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key

    # Double check that we have keys
    if not os.environ.get("HUGGINGFACEHUB_API_TOKEN") or not os.environ.get("GEMINI_API_KEY"):
        error_msg = "Missing API keys (HuggingFace token or Gemini key)."
        logger.error(f"Task {task_id} failed: {error_msg}")
        update_task_result(task_id, status="FAILED", error_message=error_msg)
        return

    try:
        logger.info(f"Task {task_id}: Building LangGraph workflow...")
        workflow = build_graph()
        
        logger.info(f"Task {task_id}: Invoking LangGraph workflow...")
        final_state = workflow.invoke({"query": query, "task_id": task_id})
        
        # Get results
        plan = final_state.get("plan", "")
        report = final_state.get("report", "")
        retrieved_papers = final_state.get("retrieved_papers", [])
        raw_papers = final_state.get("raw_papers", [])
        
        # Local evaluation
        logger.info(f"Task {task_id}: Running evaluation...")
        update_task_status(task_id, status="EVALUATING", progress_msg="Running evaluation metrics...")
        
        contexts = [p.get("abstract", "") for p in retrieved_papers]
        analysis = final_state.get("analysis", "")
        ground_truth = "Dummy truth for evaluation since query is open-ended."
        
        metrics = evaluate_pipeline(query, analysis, contexts, ground_truth)
        
        # Save complete result
        logger.info(f"Task {task_id}: Completed successfully.")
        update_task_result(
            task_id, 
            status="COMPLETED", 
            plan=plan, 
            report=report, 
            retrieved_papers=retrieved_papers, 
            raw_papers=raw_papers, 
            relationships=[],
            metrics=metrics
        )
        
    except Exception as e:
        logger.exception(f"Error during execution of task {task_id}")
        update_task_result(task_id, status="FAILED", error_message=str(e))

@app.post("/api/research/submit")
def submit_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """Submits a research topic and runs the agent pipeline in the background."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    task_id = str(uuid.uuid4())
    create_task(task_id, request.query)
    
    # Enqueue task to background threads
    background_tasks.add_task(
        run_agent_pipeline, 
        task_id, 
        request.query, 
        request.hf_token, 
        request.gemini_key
    )
    
    return {"task_id": task_id, "status": "PENDING", "message": "Research job submitted successfully."}

@app.get("/api/research/status/{task_id}")
def get_research_status(task_id: str):
    """Returns the details and progress of a research job."""
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.get("/api/research/history")
def get_research_history():
    """Returns a list of all research tasks."""
    return get_all_tasks()

@app.delete("/api/research/delete/{task_id}")
def delete_research_task(task_id: str):
    """Deletes a research task from the database."""
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    delete_task(task_id)
    return {"status": "SUCCESS", "message": f"Task {task_id} deleted successfully."}

@app.post("/api/research/notes/{task_id}")
def update_research_notes(task_id: str, request: NotesRequest):
    """Saves custom user-written research notes for a task."""
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    update_task_notes(task_id, request.notes)
    return {"status": "SUCCESS", "message": "Notes updated successfully."}

@app.get("/api/research/stats")
def get_dashboard_stats():
    """Returns backend dashboard diagnostic metrics."""
    try:
        return get_system_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute stats: {str(e)}")
