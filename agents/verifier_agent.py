import os
from langchain_google_genai import ChatGoogleGenerativeAI

def verify_report(query: str, report: str, papers: list[dict]) -> str:
    """
    Cross-references the generated report against original source paper abstracts
    to identify unsupported claims or hallucinated facts.
    """
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0.0,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    
    context = ""
    for idx, paper in enumerate(papers, 1):
        context += f"[{idx}] Title: {paper.get('title')}\nAbstract: {paper.get('abstract')}\n\n"
        
    prompt = f"""You are a professional medical safety and verification agent.
Your task is to carefully review the generated medical report and cross-reference all clinical claims against the provided source paper abstracts.

Identify any statements or treatment details in the report that are:
1. Hallucinated or completely unsupported by the provided abstracts.
2. Contradictory to the statements in the abstracts.
3. Exaggerating the results or efficacy compared to the abstracts.

If you find unsupported or contradictory claims, write a bulleted list explaining them clearly under a warning section.
If all claims in the report are fully supported by the provided abstracts, write: "All claims in this report have been successfully verified against the source literature."

Query: {query}

Report to Verify:
{report}

Source Paper Abstracts:
{context}

Format your output exactly as follows:

### 🛡️ Medical Verification & Safety Seal
[Your analysis: either the warning list, or the verification success message]
"""
    try:
        response = llm.invoke(prompt)
        verification_content = response.content.strip()
        
        # Append the verification seal to the original report
        return f"{report}\n\n{verification_content}"
    except Exception as e:
        return f"{report}\n\n### 🛡️ Medical Verification & Safety Seal\nVerification failed to run due to an error: {str(e)}"
