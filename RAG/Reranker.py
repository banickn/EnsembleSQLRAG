from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Dict, Any


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank a list of candidates using a ColBERT-style model.

        Args:
            query (str): The query to compare candidates against.
            candidates (List[Dict[str, Any]]): List of candidate items with fields to rerank.

        Returns:
            List[Dict[str, Any]]: Reranked candidates.
        """
        if not candidates:
            raise ValueError("No candidates provided for reranking.")
        inputs = [f"Query: {query} Document: {
            candidate.get('text', '')}" for candidate in candidates]
        print(f"Inputs: {inputs}")

        # Check for empty or invalid inputs
        if not any(inputs):
            raise ValueError("All input candidates are empty.")
        inputs = [f"Query: {query} Document: {
            candidate['text']}" for candidate in candidates]
        encodings = self.tokenizer(
            inputs, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            scores = self.model(**encodings).logits.squeeze(-1)
        # Attach scores to candidates
        for candidate, score in zip(candidates, scores):
            candidate["score"] = score.item()
        # Sort by score in descending order
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates
