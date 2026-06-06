import os
import numpy as np
from retrieval.embedding_model import get_embedding_model

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

def get_jaccard_similarity(str1, str2):
    s1 = set(str1.lower().split())
    s2 = set(str2.lower().split())
    if not s1 or not s2:
        return 0.0
    return len(s1.intersection(s2)) / len(s1.union(s2))

def evaluate_pipeline(query: str, answer: str, contexts: list[str], ground_truth: str = "N/A"):
    """
    Evaluates the RAG pipeline using local/embedding-based metrics.
    """
    try:
        model = get_embedding_model()
        
        # Embed query, answer, and combined contexts
        query_emb = np.array(model.embed_query(query))
        answer_emb = np.array(model.embed_query(answer))
        
        context_str = " ".join(contexts)
        if context_str.strip():
            context_emb = np.array(model.embed_query(context_str))
            faithfulness = cosine_similarity(answer_emb, context_emb)
            context_precision = cosine_similarity(query_emb, context_emb)
            lexical_overlap = get_jaccard_similarity(answer, context_str)
        else:
            faithfulness = 0.0
            context_precision = 0.0
            lexical_overlap = 0.0
            
        answer_relevancy = cosine_similarity(query_emb, answer_emb)
        
        return {
            "faithfulness": round(max(0.0, min(1.0, faithfulness)), 3),
            "answer_relevancy": round(max(0.0, min(1.0, answer_relevancy)), 3),
            "context_precision": round(max(0.0, min(1.0, context_precision)), 3),
            "context_recall": round(max(0.0, min(1.0, lexical_overlap)), 3)
        }
    except Exception as e:
        # Fallback to pure lexical Jaccard metrics if Hugging Face API has issues
        try:
            context_str = " ".join(contexts)
            ans_ctx_sim = get_jaccard_similarity(answer, context_str)
            qry_ans_sim = get_jaccard_similarity(query, answer)
            qry_ctx_sim = get_jaccard_similarity(query, context_str)
            
            return {
                "faithfulness": round(ans_ctx_sim, 3),
                "answer_relevancy": round(qry_ans_sim, 3),
                "context_precision": round(qry_ctx_sim, 3),
                "context_recall": round(ans_ctx_sim, 3),
                "warning": f"Embedding fallback triggered: {str(e)}"
            }
        except Exception as fallback_err:
            return {"error": f"Evaluation exception: {str(e)} | Fallback error: {str(fallback_err)}"}
