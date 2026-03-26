import os
from langchain_community.vectorstores import FAISS
from retrieval.embedding_model import get_embedding_model
from langchain_core.documents import Document

DB_PATH = "data/faiss_index"

class FAISSRetriever:
    def __init__(self):
        self.embeddings = get_embedding_model()
        self.vector_store = None
        self._load_or_create()

    def _load_or_create(self):
        if os.path.exists(DB_PATH):
            try:
                self.vector_store = FAISS.load_local(
                    DB_PATH, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception:
                # If loading fails, we'll recreate later when adding documents
                self.vector_store = None
        
    def add_documents(self, documents: list[Document]):
        if not documents:
            return

        if self.vector_store is None:
             self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
             self.vector_store.add_documents(documents)
             
        # Create dir if not exists
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self.vector_store.save_local(DB_PATH)

    def search(self, query: str, k: int = 3):
        if not self.vector_store:
            return []
        return self.vector_store.similarity_search(query, k=k)

    def search_with_score(self, query: str, k: int = 3):
        if not self.vector_store:
            return []
        return self.vector_store.similarity_search_with_score(query, k=k)
