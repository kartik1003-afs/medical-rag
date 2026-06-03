import sqlite3
import json
import os
from datetime import datetime

# Database is saved in the data directory
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "medical_research.db"))

def init_db():
    """Initializes the database and creates tasks table if it does not exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            query TEXT,
            status TEXT,
            progress_msg TEXT,
            plan TEXT,
            report TEXT,
            retrieved_papers TEXT,
            raw_papers TEXT,
            relationships TEXT,
            metrics TEXT,
            error_message TEXT,
            notes TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    conn.commit()
    
    # Run migrations: safely add notes column to existing database
    try:
        cursor.execute("ALTER TABLE tasks ADD COLUMN notes TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists
        
    conn.close()

def get_db_connection():
    """Returns a connection to the SQLite database."""
    return sqlite3.connect(DB_PATH)

def create_task(task_id: str, query: str):
    """Inserts a new task in PENDING state."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO tasks (id, query, status, progress_msg, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (task_id, query, "PENDING", "Task submitted and queued...", now, now))
    conn.commit()
    conn.close()

def update_task_status(task_id: str, status: str, progress_msg: str):
    """Updates the task execution phase and descriptive message."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute("""
        UPDATE tasks
        SET status = ?, progress_msg = ?, updated_at = ?
        WHERE id = ?
    """, (status, progress_msg, now, task_id))
    conn.commit()
    conn.close()

def update_task_result(task_id: str, status: str, plan: str = None, report: str = None, 
                       retrieved_papers: list = None, raw_papers: list = None, 
                       relationships: list = None, metrics: dict = None, error_message: str = None):
    """Updates the final results of the task execution (e.g. on SUCCESS or FAILURE)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    
    retrieved_papers_json = json.dumps(retrieved_papers) if retrieved_papers is not None else None
    raw_papers_json = json.dumps(raw_papers) if raw_papers is not None else None
    relationships_json = json.dumps(relationships) if relationships is not None else None
    metrics_json = json.dumps(metrics) if metrics is not None else None
    
    updates = [("status", status), ("updated_at", now)]
    if plan is not None:
        updates.append(("plan", plan))
    if report is not None:
        updates.append(("report", report))
    if retrieved_papers_json is not None:
        updates.append(("retrieved_papers", retrieved_papers_json))
    if raw_papers_json is not None:
        updates.append(("raw_papers", raw_papers_json))
    if relationships_json is not None:
        updates.append(("relationships", relationships_json))
    if metrics_json is not None:
        updates.append(("metrics", metrics_json))
    if error_message is not None:
        updates.append(("error_message", error_message))
        
    query_str = f"UPDATE tasks SET {', '.join([f'{col} = ?' for col, _ in updates])} WHERE id = ?"
    values = [val for _, val in updates] + [task_id]
    
    cursor.execute(query_str, values)
    conn.commit()
    conn.close()

def get_task(task_id: str) -> dict:
    """Fetches a single task by ID and deserializes JSON columns."""
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
        
    task = dict(row)
    # Deserialize JSON fields
    if task.get("retrieved_papers"):
        task["retrieved_papers"] = json.loads(task["retrieved_papers"])
    if task.get("raw_papers"):
        task["raw_papers"] = json.loads(task["raw_papers"])
    if task.get("relationships"):
        task["relationships"] = json.loads(task["relationships"])
    if task.get("metrics"):
        task["metrics"] = json.loads(task["metrics"])
    return task

def get_all_tasks() -> list:
    """Fetches all tasks sorted by creation date."""
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tasks ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    
    tasks = []
    for row in rows:
        task = dict(row)
        if task.get("retrieved_papers"):
            task["retrieved_papers"] = json.loads(task["retrieved_papers"])
        if task.get("raw_papers"):
            task["raw_papers"] = json.loads(task["raw_papers"])
        if task.get("relationships"):
            task["relationships"] = json.loads(task["relationships"])
        if task.get("metrics"):
            task["metrics"] = json.loads(task["metrics"])
        tasks.append(task)
    return tasks

def delete_task(task_id: str):
    """Deletes a research task from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()

def update_task_notes(task_id: str, notes: str):
    """Saves custom user-written research notes for a task."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute("""
        UPDATE tasks
        SET notes = ?, updated_at = ?
        WHERE id = ?
    """, (notes, now, task_id))
    conn.commit()
    conn.close()

def get_system_stats() -> dict:
    """Queries SQL to compute backend dashboard diagnostic metrics."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Total jobs run
    cursor.execute("SELECT COUNT(*) FROM tasks")
    total_jobs = cursor.fetchone()[0]
    
    # 2. Completed / Failed breakdown
    cursor.execute("SELECT COUNT(*) FROM tasks WHERE status = 'COMPLETED'")
    completed_jobs = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM tasks WHERE status = 'FAILED'")
    failed_jobs = cursor.fetchone()[0]
    
    # 3. Popular queries (top 5 keywords/queries)
    cursor.execute("SELECT query, COUNT(*) as cnt FROM tasks GROUP BY query ORDER BY cnt DESC LIMIT 5")
    popular_queries = [{"query": row[0], "count": row[1]} for row in cursor.fetchall()]
    
    conn.close()
    
    success_rate = (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0.0
    
    return {
        "total_jobs": total_jobs,
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
        "success_rate": round(success_rate, 1),
        "popular_queries": popular_queries
    }
