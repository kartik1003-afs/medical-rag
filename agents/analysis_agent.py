import os
from langchain_google_genai import ChatGoogleGenerativeAI

def analyze_papers(query: str, papers: list[dict]) -> str:
    """
    Analyzes the retrieved papers and extracts key findings related to the query.
    Utilizes Gemini 2.5 Flash model.
    """
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0.0,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    
    context = ""
    for idx, paper in enumerate(papers, 1):
        context += f"[{idx}] Title: {paper.get('title')}\nAbstract: {paper.get('abstract')}\n\n"
        
    prompt = f"""You are a professional medical research assistant. Based on the following research papers, extract key findings related to the query. 
If the query pertains to a disease, you MUST extract and categorize specific "Treatment Details", including:
1. Treatment Names/Methods
2. Efficacy & Results
3. Noted Side Effects (if any)

Query: {query}

Research Papers:
{context}

Format your response exactly as follows:
### Core Findings
[A general structured summary]

### Treatment Details
[Detailed markdown bullet points or tables highlighting treatments, their efficacy, and side effects]
"""
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error analyzing papers: {str(e)}"
