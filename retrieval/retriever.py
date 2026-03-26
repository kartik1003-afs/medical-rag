from retrieval.faiss_store import FAISSRetriever
from langchain_core.documents import Document

class DocumentRetriever:
    def __init__(self):
        self.db = FAISSRetriever()

    def add_papers_to_index(self, papers: list):
        """
        Takes a list of papers (dicts containing title, abstract, pmid, etc.)
        and indexes them into FAISS.
        """
        documents = []
        for paper in papers:
            content = f"Title: {paper.get('title', '')}\nAuthors: {paper.get('authors', '')}\nAbstract: {paper.get('abstract', '')}\nYear: {paper.get('year', '')}"
            doc = Document(page_content=content, metadata=paper)
            documents.append(doc)
            
        self.db.add_documents(documents)

    def get_relevant_papers(self, query: str, top_k: int = 3):
        results = self.db.search_with_score(query, k=top_k)
        papers = []
        for doc, score in results:
            paper_meta = doc.metadata.copy()
            # FAISS distance metric: lower score = more similar. Convert to simple normalized or just store crude relative float
            paper_meta['similarity_score'] = round(float(score), 4)
            papers.append(paper_meta)
        return papers
