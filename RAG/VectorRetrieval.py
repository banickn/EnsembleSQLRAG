import os
import json
import logging
from typing import List, Optional, Dict, Union
from pathlib import Path

import lancedb
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from enum import Enum, auto
from functools import lru_cache


class VectorDataType(Enum):
    METADATA = auto()
    QUERYDATA = auto()


class VectorRetrieval:
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        vector_db_path: str = "./lancedb_data",
        vector_column: str = "vector_embedding",
        data_type: VectorDataType = VectorDataType.METADATA,
        cache_size: int = 128
    ):
        """
        Advanced Vector Retrieval System with enhanced configurability.

        Args:
            embedding_model (str): Sentence transformer model
            vector_db_path (str): Path for vector database storage
            vector_column (str): Name of embedding column
            data_type (VectorDataType): Type of vector data
            cache_size (int): LRU cache size for embeddings
        """
        # Load environment variables

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        self.embedding_model = SentenceTransformer(embedding_model)
        self.vector_db_path = Path(vector_db_path)
        self.vector_db_path.mkdir(parents=True, exist_ok=True)

        self.vector_column = vector_column
        self.data_type = data_type

        self.connection = lancedb.connect(str(self.vector_db_path))
        self.table = None

    def generate_embeddings(
        self,
        data: pd.DataFrame,
        embedding_columns: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Cached embedding generation with enhanced error handling.

        Args:
            data (pd.DataFrame): Input dataframe
            embedding_columns (List[str], optional): Columns for embedding

        Returns:
            np.ndarray: Generated embeddings
        """
        try:
            embedding_columns = embedding_columns or list(data.columns)

            if not set(embedding_columns).issubset(data.columns):
                raise ValueError("Invalid embedding columns specified")

            text_series = data[embedding_columns].apply(
                lambda row: " ".join(row.astype(str)),
                axis=1
            )

            embeddings = self.embedding_model.encode(
                text_series.tolist(),
                show_progress_bar=False,
                convert_to_numpy=True
            )

            return embeddings

        except Exception as e:
            self.logger.error(f"Embedding generation error: {e}")
            return np.array([])

    def load_data(
        self,
        json_path: Union[str, Path]
    ) -> List[Dict]:
        """
        Unified data loading with enhanced error handling.

        Args:
            json_path (Union[str, Path]): Path to JSON data

        Returns:
            List[Dict]: Parsed data objects
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            if self.data_type == VectorDataType.METADATA:
                return [
                    {
                        "table_name": table["table_name"],
                        "column_name": column["column_name"],
                        "column_type": column["column_type"],
                        "column_remarks": column.get("column_remarks", "")
                    }
                    for table in data.get("tables", [])
                    for column in table.get("columns", [])
                ]

            elif self.data_type == VectorDataType.QUERYDATA:
                return data.get("queries", [])

        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Data parsing error: {e}")
            return []

    def ingest_data(
        self,
        json_path: Union[str, Path],
        table_name: str = "documents",
        embedding_columns: Optional[List[str]] = None,
        overwrite: bool = True
    ) -> None:
        """
        Ingest data into LanceDB with robust error handling and flexibility.

        Args:
            json_path (Union[str, Path]): Path to JSON data source
            table_name (str): Name of the vector database table
            embedding_columns (List[str], optional): Columns for embedding generation
            overwrite (bool): Whether to overwrite existing table
        """
        try:
            # Load data based on type
            data = self.load_data(json_path)

            if not data:
                self.logger.warning(f"No data found in {json_path}")
                return

            # Convert to DataFrame
            data_df = pd.DataFrame(data)

            # Generate embeddings
            embeddings = self.generate_embeddings(data_df, embedding_columns)

            if len(embeddings) == 0:
                self.logger.error("Failed to generate embeddings")
                return

            # Prepare LanceDB data
            lancedb_data = data_df.to_dict(orient='records')
            for i, item in enumerate(lancedb_data):
                item[self.vector_column] = embeddings[i].tolist()

            # Ingest into LanceDB
            ingestion_mode = "overwrite" if overwrite else "append"
            self.table = self.connection.create_table(
                table_name,
                data=lancedb_data,
                mode=ingestion_mode
            )

            self.logger.info(f"Successfully ingested {len(data)} records")

        except Exception as e:
            self.logger.error(f"Data ingestion failed: {e}")

    def semantic_search(
        self,
        query: str,
        table_name: str = "documents",
        top_k: int = 10,
        similarity_threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Enhanced semantic search with similarity threshold.

        Args:
            query (str): Search query
            table_name (str): Target table
            top_k (int): Maximum results
            similarity_threshold (float): Minimum similarity score

        Returns:
            pd.DataFrame: Filtered search results
        """
        try:
            query_embedding = self.embedding_model.encode(query)

            if self.table is None:
                self.table = self.connection.open_table(table_name)

            results = self.table.search(
                query_embedding).limit(top_k).to_pandas()

            # Optional: Filter by similarity threshold
            results = results[results['_distance'] <= similarity_threshold]
            return results

        except Exception as e:
            self.logger.error(f"Semantic search error: {e}")
            return pd.DataFrame()
