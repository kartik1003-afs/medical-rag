def generate_report(query: str, analysis: str, papers: list[dict]) -> str:
    """
    Generates a structured research summary from the analysis and retrieved papers.
    """
    report = f"## Topic: {query}\n\n"
    report += "### Key Findings\n"
    report += f"{analysis}\n\n"
    
    report += "### Sources\n"
    for paper in papers:
        pmid = paper.get('pmid', 'N/A')
        title = paper.get('title', 'Unknown Title')
        if pmid and pmid.strip():
            report += f"- **PubMed ID:** {pmid} - {title}\n"
        else:
            report += f"- {title}\n"
            
    return report
