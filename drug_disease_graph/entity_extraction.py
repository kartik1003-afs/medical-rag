import os
from langchain_google_genai import ChatGoogleGenerativeAI
import json

def extract_entities(abstract: str) -> dict:
    """Extracts 'Disease' and 'Drug' entities from a medical abstract."""
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0.0,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    prompt = f"""Extract 'Disease' and 'Drug' entities from the abstract.
Return ONLY valid JSON format with keys 'diseases' and 'drugs' as lists of strings.
Abstract: {abstract}
JSON output:"""
    try:
        j_str = llm.invoke(prompt).content
        j_str = j_str.replace("```json", "").replace("```", "").strip()
        return json.loads(j_str)
    except:
        return {"diseases": [], "drugs": []}
