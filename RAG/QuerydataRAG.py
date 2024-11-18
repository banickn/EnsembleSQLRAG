from RAG.RAGBase import RAGBase
from RAG.QueryData import QueryData
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from typing import List, Dict, Any, Tuple, Optional, TypeVar, Generic
import logging


class QuerydataRAG(RAGBase[QueryData]):
    """RAG implementation for database column querydata"""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        weights: Optional[Dict[str, float]] = None
    ):
        weights = weights or {
            'user': 0.1,
            'query': 0.9,
            'tables': 0.6
        }
        super().__init__(model_name, weights)

    def _initialize_indices(self):
        """Initialize indices for column querydata components"""
        components = ['user', 'query', 'tables']
        for component in components:
            self.indices[component] = faiss.IndexFlatL2(self.dimension)

    def _create_querydata_from_dict(self, data: Dict[str, Any]) -> QueryData:
        return QueryData.from_dict(data)

    def load_json_querydata(self, json_path: str) -> List[QueryData]:
        """Load querydata from JSON file"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            query_list = []
            for query in data:
                querydata = QueryData(
                    user=query["user"],
                    query=query['query'],
                    tables=query['tables'],
                )
                query_list.append(querydata)
            print(query_list)
            return query_list

        except Exception as e:
            logging.error(f"Error loading JSON querydata: {e}")
            return []
