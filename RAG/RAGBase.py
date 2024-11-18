from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from typing import List, Dict, Any, Tuple, Optional, TypeVar, Generic
from dataclasses import dataclass
import logging
from pathlib import Path
T = TypeVar('T')


class RAGBase(Generic[T]):
    """Base class for RAG implementations"""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        weights: Optional[Dict[str, float]] = None
    ):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.data_store: List[T] = []
        self.indices: Dict[str, Any] = {}
        self.weights = weights or {}

        # Initialize FAISS indices
        self._initialize_indices()

    @abstractmethod
    def _initialize_indices(self):
        """Initialize FAISS indices based on data components"""
        pass

    def add_data(self, data_list: List[T]):
        """Add data to the RAG system"""
        if not data_list:
            logging.warning("No data provided to add")
            return

        self.data_store.extend(data_list)

        # Get texts for each component
        component_texts = {
            component: [] for component in self.indices.keys()
        }

        # Collect and validate texts
        for data in data_list:
            try:
                texts = data.to_text()
                if not isinstance(texts, dict):
                    raise ValueError(
                        f"to_text() must return a dictionary, got {type(texts)}")

                for component, text in texts.items():
                    if component not in self.indices:
                        logging.warning(
                            f"Skipping unknown component: {component}")
                        continue

                    if not isinstance(text, str):
                        logging.warning(
                            f"Skipping non-string text for component {component}: {type(text)}")
                        continue

                    if not text.strip():
                        logging.warning(
                            f"Skipping empty text for component {component}")
                        continue

                    component_texts[component].append(text)

            except Exception as e:
                logging.error(f"Error processing data item: {e}")
                continue

        # Generate embeddings and add to indices
        for component, texts in component_texts.items():
            if not texts:
                logging.warning(f"No valid texts for component {component}")
                continue

            try:
                logging.debug(
                    f"Encoding {len(texts)} texts for component {component}")
                embeddings = self.model.encode(texts, show_progress_bar=False)
                self.indices[component].add(
                    np.array(embeddings).astype('float32'))
            except Exception as e:
                logging.error(f"Error encoding texts for component {
                              component}: {e}")

    def search(
        self,
        query: str,
        k: int = 5,
        custom_weights: Optional[Dict[str, float]] = None
    ) -> List[Tuple[T, float]]:
        """Search for relevant data"""
        weights = custom_weights or self.weights

        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {
            component: weight / total_weight
            for component, weight in weights.items()
        }

        # Generate query embedding
        query_embedding = self.model.encode([query])

        # Search in each index
        component_results = {}
        for component, index in self.indices.items():
            distances, indices = index.search(
                np.array(query_embedding).astype('float32'), k
            )
            component_results[component] = {
                idx: float(dist)
                for idx, dist in zip(indices[0], distances[0])
                if idx < len(self.data_store)
            }

        # Combine scores
        combined_scores = {}
        for idx in range(len(self.data_store)):
            score = 0.0
            for component, weight in normalized_weights.items():
                if idx in component_results[component]:
                    similarity = 1 / (1 + component_results[component][idx])
                    score += weight * similarity
            if score > 0:
                combined_scores[idx] = score

        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        return [(self.data_store[idx], score) for idx, score in sorted_results]

    def save_index(self, path: str):
        """Save indices and data"""
        path = Path(path)

        for component, index in self.indices.items():
            component_path = path.with_name(
                f"{path.stem}_{component}").with_suffix('.index')
            faiss.write_index(index, str(component_path))

        save_data = {
            'weights': self.weights,
            'data': [m.to_dict() for m in self.data_store]
        }

        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(save_data, f, indent=2)

    def load_index(self, path: str):
        """Load indices and data"""
        path = Path(path)

        for component in self.indices.keys():
            component_path = path.with_name(
                f"{path.stem}_{component}").with_suffix('.index')
            self.indices[component] = faiss.read_index(str(component_path))

        with open(path.with_suffix('.json'), 'r') as f:
            save_data = json.load(f)
            self.weights = save_data['weights']
            self.data_store = [
                self._create_data_from_dict(m) for m in save_data['data']
            ]

    @abstractmethod
    def _create_data_from_dict(self, data: Dict[str, Any]) -> T:
        """Create data object from dictionary"""
        pass
