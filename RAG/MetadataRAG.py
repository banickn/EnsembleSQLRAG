from RAG.RAGBase import RAGBase
from RAG.ColumnMetadata import ColumnMetadata
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from typing import List, Dict, Any, Tuple, Optional, TypeVar, Generic
import logging


class MetadataRAG(RAGBase[ColumnMetadata]):
    """RAG implementation for database column metadata"""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        weights: Optional[Dict[str, float]] = None
    ):
        weights = weights or {
            'name': 0.4,
            'type': 0.2,
            'remarks': 0.3,
            'context': 0.6
        }
        super().__init__(model_name, weights)

    def _initialize_indices(self):
        """Initialize indices for column metadata components"""
        components = ['name', 'type', 'remarks', 'context']
        for component in components:
            self.indices[component] = faiss.IndexFlatL2(self.dimension)

    def _create_metadata_from_dict(self, data: Dict[str, Any]) -> ColumnMetadata:
        return ColumnMetadata.from_dict(data)

    def load_json_metadata(self, json_path: str) -> List[ColumnMetadata]:
        """Load metadata from JSON file"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            metadata_list = []
            for table in data["tables"]:
                for column_info in table["columns"]:
                    metadata = ColumnMetadata(
                        table_name=table["table_name"],
                        column_name=column_info['column_name'],
                        column_type=column_info['column_type'],
                        column_remarks=column_info['column_remarks']
                    )
                    metadata_list.append(metadata)

            return metadata_list

        except Exception as e:
            logging.error(f"Error loading JSON metadata: {e}")
            return []
