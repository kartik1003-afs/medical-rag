import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings

def get_embedding_model():
    """
    Returns the Hugging Face embedding model.
    """
    return HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-small-en-v1.5",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )
