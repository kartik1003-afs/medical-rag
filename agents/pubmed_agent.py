import requests
import xml.etree.ElementTree as ET

def search_pubmed(query: str, max_results: int = 5) -> list[dict]:
    """
    Search PubMed for research papers using the E-utilities API.
    Returns a list of dictionaries containing paper details.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    # 1. Search for IDs
    search_url = f"{base_url}/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json"
    }
    
    response = requests.get(search_url, params=search_params)
    response.raise_for_status()
    data = response.json()
    id_list = data.get("esearchresult", {}).get("idlist", [])
    
    if not id_list:
        return []
        
    # 2. Fetch details for these IDs
    fetch_url = f"{base_url}/efetch.fcgi"
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(id_list),
        "retmode": "xml"
    }
    
    fetch_response = requests.get(fetch_url, params=fetch_params)
    fetch_response.raise_for_status()
    
    # Parse XML
    root = ET.fromstring(fetch_response.content)
    papers = []
    
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID")
        title = article.findtext(".//ArticleTitle")
        
        abstract_element = article.find(".//Abstract")
        abstract = ""
        if abstract_element is not None:
            abstract_texts = abstract_element.findall(".//AbstractText")
            abstract = " ".join([t.text for t in abstract_texts if t.text])
            
        pub_date = article.find(".//PubDate")
        year = pub_date.findtext(".//Year") if pub_date is not None else ""
        
        journal_element = article.find(".//Journal/Title")
        journal = journal_element.text if journal_element is not None else "Unknown Journal"
        
        authors_list = []
        for author in article.findall(".//Author"):
            last_name = author.findtext("LastName") or ""
            initials = author.findtext("Initials") or ""
            if last_name:
                authors_list.append(f"{last_name} {initials}".strip())
                
        papers.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "year": year,
            "journal": journal,
            "authors": ", ".join(authors_list),
            "similarity_score": 0.0,
            "study_type": "Other"
        })
        
    return papers
