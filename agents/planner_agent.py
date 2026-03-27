import os
from langchain_google_genai import ChatGoogleGenerativeAI

def generate_plan(query: str) -> str:
    """
    Break down the user query into actionable subtasks.
    """
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0.0,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    
    prompt = f"""You are a medical planner agent. Break down the user query into a short list of 3-4 concise subtasks.
User Query: {query}
Plan:"""
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"1. Retrieve papers for: {query}\n2. Analyze findings\n3. Generate report"
