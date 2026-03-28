import os
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List

def classify_study_types(papers: List[dict]) -> List[dict]:
    """
    Classifies the study type of given research papers using Gemini.
    """
    if not papers:
        return papers
        
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0.0,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        timeout=60,
        max_retries=2
    )
    
    context = ""
    for idx, p in enumerate(papers):
        title = p.get('title', 'N/A')
        abstract = p.get('abstract', 'N/A')
        context += f"Index {idx} | Title: {title} | Abstract: {abstract}\n\n"
        
    prompt = f"""You are an Evidence Ranking Agent. Classify each of the following research papers into one of these exact study types:
- Randomized Clinical Trial
- Meta Analysis
- Cohort Study
- Case Study
- Review
- Other

Output ONLY a comma-separated list of classifications correlating to the Index of each paper.
Example output for 3 papers: Randomized Clinical Trial, Cohort Study, Other

Papers:
{context}

Classifications:"""
    try:
         res = llm.invoke(prompt).content
         # Split by comma and strip
         types = [t.strip() for t in res.split(',')]
         # Assign back to papers
         for i, p in enumerate(papers):
             if i < len(types):
                 # Quick validation
                 valid_types = ['Randomized Clinical Trial', 'Meta Analysis', 'Cohort Study', 'Case Study', 'Review', 'Other']
                 p['study_type'] = types[i] if any(v.lower() in types[i].lower() for v in valid_types) else 'Other'
             else:
                 p['study_type'] = 'Other'
    except Exception:
         for p in papers:
             p['study_type'] = 'Other'
             
    return papers
