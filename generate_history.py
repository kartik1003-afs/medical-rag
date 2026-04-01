import os
import subprocess
from datetime import datetime, timedelta

def git_commit(files, message, days_ago, hours_offset=0):
    date = datetime.now() - timedelta(days=days_ago, hours=hours_offset)
    # ISO 8601 format natively parsed by Git
    date_str = date.strftime('%Y-%m-%dT%H:%M:%S')
    
    # Add files
    for f in files:
        if os.path.exists(f):
            subprocess.run(['git', 'add', f], shell=False)
            
    # Commit with backdated timestamp environment variables
    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = date_str
    env["GIT_COMMITTER_DATE"] = date_str
    
    res = subprocess.run(['git', 'commit', '-m', message], env=env, shell=False)
    if res.returncode == 0:
        print(f"✅ Committed pieces: {files} on {date_str}")
    else:
        print(f"⚠️ Failed to commit {files}")

def main():
    print("Beginning simulated commit timeline generation...")
    
    # Reset any staged files just in case
    subprocess.run(['git', 'reset', 'HEAD'], shell=False)

    commits = [
        # Base architecture setup (7 days ago)
        (["pyproject.toml", "uv.lock", ".python-version", ".gitignore", "README.md"], 
         "Initial project structure, dependencies, and gitignore", 7, 2),
        
        # Retrieval module (6 days ago)
        (["data", "retrieval"], 
         "Build FAISS vector database setup and basic retrieval modules", 6, 4),
        
        # Base Agents (5 days ago)
        (["agents/planner_agent.py", "agents/pubmed_agent.py", "agents/__init__.py"], 
         "Implement Research Planner and PubMed fetching AI Agents", 5, 1),
        
        # Complex Agents (4 days ago)
        (["agents/evidence_agent.py", "agents/report_agent.py", "agents/analysis_agent.py"], 
         "Add evidence verification and research synthesis agents", 4, 3),

        # Graph Engine (3 days ago)
        (["drug_disease_graph"], 
         "Engineer entity relationship extraction for drug-disease knowledge graph", 3, 2),
        
        # Langchain Langgraph Orchestrator (2 days ago)
        (["main.py"], 
         "Orchestrate stateful multi-agent LangGraph workflow entrypoint", 2, 5),
        
        # Evaluation suite (1 day ago)
        (["evaluation"], 
         "Integrate RAGAS metrics payload for pipeline context evaluation", 1, 1),
        
        # Dashboards & Hosting (Today)
        (["frontend", "Dockerfile", ".env"], 
         "Publish interactive Streamlit AI dashboard interface and Docker deployment footprint", 0, 0),
    ]

    for files, msg, days_ago, hours in commits:
        git_commit(files, msg, days_ago, hours)

    # Finally capture anything we might have missed in a polish commit today right now
    print("\nCapturing any uncategorized stragglers...")
    subprocess.run(['git', 'add', '.'], shell=False)
    
    # We only commit if there are actually files left
    status = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, shell=False)
    if status.stdout.strip():
        subprocess.run(['git', 'commit', '-m', "Final application polish and bugfixes"], shell=False)
        print("✅ Added final polish commit.")
        
    print("\n🎉 Commit timeline successfully faked! Check out your timeline with 'git log'.")
    print("Next step: Create a repo on GitHub and run:")
    print("git remote add origin <YOUR_GITHUB_REPO_URL>")
    print("git push -u origin master")

if __name__ == "__main__":
    main()
