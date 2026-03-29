import os
from langchain_google_genai import ChatGoogleGenerativeAI
import json

def extract_relationships(abstract: str) -> list:
    """
    Extracts relationships of the form Disease -> treated_by -> Drug.
    """
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0.0,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    prompt = f"""Analyze the abstract and extract treatment relationships between Diseases and Drugs.
Only keep relationships where a Disease is extracted as 'treated_by' a Drug.
Return ONLY a valid JSON list of dictionaries with keys "disease" and "drug".
Abstract: {abstract}
JSON format:
[
  {{"disease": "DiseaseName", "drug": "DrugName"}}
]
"""
    try:
        j_str = llm.invoke(prompt).content
        j_str = j_str.replace("```json", "").replace("```", "").strip()
        return json.loads(j_str)
    except:
        return []
