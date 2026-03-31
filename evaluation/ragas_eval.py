import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings

def evaluate_pipeline(query: str, answer: str, contexts: list[str], ground_truth: str = "N/A"):
    """
    Evaluates the RAG pipeline using Ragas metrics.
    """
    evaluator_llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0.0,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        timeout=120,
        max_retries=3
    )
    
    evaluator_embeddings = HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-small-en-v1.5",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    data_sample = {
        "question": [query],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [ground_truth]
    }
    
    dataset = Dataset.from_dict(data_sample)
    
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
    
    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            raise_exceptions=False
        )
        
        # safely extract metric output
        metrics_dict = {}
        for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            try:
                metrics_dict[m] = float(result[m])
            except Exception:
                metrics_dict[m] = 0.0
        return metrics_dict
    except Exception as e:
        return {"error": f"Evaluation exception: {str(e)}"}
