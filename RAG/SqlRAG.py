import os
import json
import logging
from typing import List, Dict, Optional, Union
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

from RAG.VectorRetrieval import VectorRetrieval, VectorDataType


class SqlRAGConfig:
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        ai_model: str = 'gemini-1.5-flash',
        meta_vector_path: str = "data/meta.lance",
        queries_vector_path: str = "data/queries.lance",
        metadata_path: str = "data/input/fake_columns.json",
        queries_path: str = "data/input/fake_queries.json"
    ):
        self.embedding_model = embedding_model
        self.ai_model = ai_model
        self.meta_vector_path = meta_vector_path
        self.queries_vector_path = queries_vector_path
        self.metadata_path = metadata_path
        self.queries_path = queries_path


class SqlRAG:
    def __init__(
        self,
        config: Optional[SqlRAGConfig] = None,
        ingest: bool = False,
        use_input_variations: bool = True
    ):
        """
        Initialize SqlRAG with configurable settings.

        Args:
            config (SqlRAGConfig): Configuration object
            ingest (bool): Whether to ingest data during initialization
            use_input_variations (bool): Use AI for query variation
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config or SqlRAGConfig()
        load_dotenv()
        self._setup_ai_client()

        self.ingest = ingest
        self.use_input_variations = use_input_variations

    def _setup_ai_client(self):
        """Configure AI client with error handling."""
        try:
            genai.configure(api_key=os.getenv("gemini_api_key"))
            self.ai_model = genai.GenerativeModel(
                model_name=self.config.ai_model,
                system_instruction=["Expert SQL query generator"]
            )
        except Exception as e:
            self.logger.error(f"AI client setup failed: {e}")
            raise

    def merge_results(self, meta_results: pd.DataFrame, queries_results: pd.DataFrame, meta_weight: int = 0.5, queries_weight: int = 0.5) -> pd.DataFrame:
        merged_df = pd.DataFrame(columns=['item', 'tables', 'score', 'origin'])
        for _, row in meta_results.iterrows():
            row_df = pd.DataFrame([{
                "item": row["column_name"],
                "tables": row["table_name"],
                "score": row["_distance"]*meta_weight,
                "origin": "metadata"
            }])
            merged_df = pd.concat([merged_df, row_df], ignore_index=True)

        # Append rows from queries_results
        for _, row in queries_results.iterrows():
            row_df = pd.DataFrame([{
                "item": row["query"],
                "tables": row["tables"],
                "score": row["_distance"]*queries_weight,
                "origin": "historical_queries"
            }])
            merged_df = pd.concat([merged_df, row_df], ignore_index=True)
        return merged_df

    def rerank_results(self,
                       merged_df: pd.DataFrame,
                       original_query: str,
                       embedding_model: SentenceTransformer,
                       rerank_method: str = 'cross-encoder',
                       top_k: int = 10
                       ) -> pd.DataFrame:
        """
        Rerank semantic search results using advanced techniques.

        Args:
            merged_df (pd.DataFrame): Merged results DataFrame from merge_results
            original_query (str): Original search query
            embedding_model (SentenceTransformer): Embedding model for additional processing
            rerank_method (str): Reranking method to use
            top_k (int): Number of top results to return

        Returns:
            pd.DataFrame: Reranked and filtered results
        """
        # Validate input
        if merged_df.empty:
            return pd.DataFrame(columns=['item', 'tables', 'score', 'origin'])

        # Import cross-encoder if needed
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            print("Cross-encoder not available. Falling back to current ranking.")
            return merged_df.sort_values('score', ascending=False).head(top_k)

        # Choose reranking method
        if rerank_method == 'cross-encoder':
            # Load a cross-encoder model for more precise reranking
            cross_encoder = CrossEncoder(
                'cross-encoder/ms-marco-MiniLM-L-6-v2')

            # Prepare data for cross-encoder
            rerank_data = [(original_query, row['item'])
                           for _, row in merged_df.iterrows()]

            # Get cross-encoder scores
            cross_encoder_scores = cross_encoder.predict(rerank_data)

            # Add cross-encoder scores to the DataFrame
            merged_df['rerank_score'] = cross_encoder_scores

            # Combine original semantic similarity with cross-encoder score
            merged_df['final_score'] = (
                merged_df['score'] * 0.6 +
                merged_df['rerank_score'] * 0.4
            )

            # Sort and return top results
            return (merged_df.sort_values('final_score', ascending=False)
                    .drop_duplicates(subset=['item'])
                    .head(top_k))

        elif rerank_method == 'similarity_fusion':
            # Advanced similarity fusion approach
            query_embedding = embedding_model.encode(original_query)

            # Calculate additional similarity scores
            def calculate_similarity(text):
                text_embedding = embedding_model.encode(text)
                return np.dot(query_embedding, text_embedding) / (
                    np.linalg.norm(query_embedding) *
                    np.linalg.norm(text_embedding)
                )

            merged_df['fusion_score'] = merged_df['item'].apply(
                calculate_similarity)

            # Combine multiple similarity metrics
            merged_df['final_score'] = (
                merged_df['score'] * 0.5 +
                merged_df['fusion_score'] * 0.5
            )

            return (merged_df.sort_values('final_score', ascending=False)
                    .drop_duplicates(subset=['item'])
                    .head(top_k))

        else:
            # Default: sort by original score
            return merged_df.sort_values('score', ascending=False).head(top_k)

    def create_ddl_from_json(self, table_data, tablename):
        """
        Create DDL statement from JSON table definition.

        Args:
            table_data (dict): Dictionary containing table definitions
            tablename (str): Name of the table to generate DDL for

        Returns:
            str: DDL statement for creating the table
        """
        # Find the specific table definition
        table_definition = next(
            (table for table in table_data["tables"] if table["table_name"] == tablename), None)

        if not table_definition:
            raise ValueError(
                f"Table {tablename} not found in the provided JSON")

        # Start building the CREATE TABLE statement
        ddl = f"CREATE TABLE {tablename} (\n"

        # Process each column
        column_definitions = []
        for column in table_definition["columns"]:
            # Build column definition
            column_def = f"    {column['column_name']} {column['column_type']}"

            # Add comment if it exists and is not null
            if column.get('column_remarks'):
                column_def += f" COMMENT '{column['column_remarks']}'"

            column_definitions.append(column_def)

        # Add primary key constraint if defined
        if table_definition.get('primary_key'):
            primary_keys = table_definition['primary_key']
            column_definitions.append(
                f"    PRIMARY KEY ({', '.join(primary_keys)})")

        # Add foreign key constraints if defined
        if table_definition.get('foreign_keys'):
            for fk in table_definition['foreign_keys']:
                fk_def = (f"    FOREIGN KEY ({fk['column_name']}) "
                          f"REFERENCES {fk['references_table']}({fk['references_column']})")
                column_definitions.append(fk_def)

        # Join all column definitions with commas
        ddl += ",\n".join(column_definitions)

        # Close the CREATE TABLE statement
        ddl += "\n);"

        return ddl

    def get_sql_recommendations(self, query: str, reranked_results: pd.DataFrame, table_ddl: list[str]) -> str:
        """
        Use Google Generative AI to get SQL recommendations.

        Parameters:
            prompt (str): A natural language prompt describing the SQL query or recommendations you want.

        Returns:
            str: The recommended SQL query.
        """
        prompt = f"""

        # SQL Query Generation Template
        ## User Input Keywords
        {query}

        ## Reranked Context
        {reranked_results}

        ## Schema Context
        ```sql
        {' '.join([table for table in table_ddl])}
        ```

        ## Query Requirements Template
        Please provide the following information to generate an optimized SQL query:

        1. **Primary Objective**
        - What is the main metric or insight you want to calculate based on the keywords?
        - Example: "Find active users based on order frequency"

        2. **Ranked Context**
        - Use the given Reranked Context to construct the query

        3. **Key Metrics**
        - List specific metrics needed in the output
        - Example: order count, total spend, average order value

        4. **Time Period**
        - Specify the time range for the analysis
        - Example: last 30 days, year-to-date, all time

        5. **Filtering Criteria**
        - Any specific conditions to filter the data?
        - Example: minimum order count, minimum spend

        6. **Output Format**
        - How should results be ordered/ranked?
        - What's the desired limit of results?

        ## Example Query Request
        ```
        Keywords: active users orders rank
        Primary Objective: Rank users by order activity
        Time Period: Year-to-date (2024)
        Metrics Needed: Username, order count, total spend
        Filtering: Only users with at least 1 order
        Ranking: By order count descending
        ```

        ## Generated Query Structure
        ```sql
        WITH UserMetrics AS (
            SELECT
                u.user_id,
                u.username,
                COUNT(DISTINCT o.order_id) as order_count,
                SUM(o.total_amount) as total_spend
            FROM users u
            JOIN orders o ON u.user_id = o.user_id
            WHERE o.order_date >= '2024-01-01'
            GROUP BY u.user_id, u.username
            HAVING order_count >= 1
        )
        SELECT
            username,
            order_count,
            total_spend,
            RANK() OVER (ORDER BY order_count DESC) as activity_rank
        FROM UserMetrics
        ORDER BY activity_rank;
        ```

        ## Best Practices
        1. Use CTEs for complex queries to improve readability
        2. Include appropriate JOINs based on foreign key relationships given in the table schemas. DO NOT use joins without a specified foreign key.
        3. Add specific WHERE clauses for time-based filtering
        4. Use window functions for ranking when needed
        5. Include HAVING clauses for aggregate filters
        6. Specify ORDER BY for clear result presentation

        """
        try:
            # Call the model with the prompt
            model = genai.GenerativeModel(model_name='gemini-1.5-flash',
                                          system_instruction=["You're an expert SQL domain expert. You are tasked with generating SQL queries based on the user's input. Your responses should be in the form of SQL statements, not natural language. You should only respond with the pure SQL code and no text. Always also select the primary key column."])
            response = model.generate_content(prompt)
            # Extract and return the first response
            return response.text if response and response.text else "No recommendation generated."
        except Exception as e:
            return f"An error occurred: {e}"

    def get_input_variations(self, query: str) -> str:
        try:
            # Call the model with the prompt
            model = genai.GenerativeModel(model_name='gemini-1.5-flash',
                                          system_instruction=["You are a helpful assistant skilled in extracting key terms from questions."])
            keyword_prompt = f"""
            Analyze the following user question and extract 3–5 key terms or phrases that best capture its essence:
            User Question: {query}
            Focus on nouns, key concepts, and any domain-specific terminology.
            Also add some variations of the key words to overcome problems with similarity search. Separate word with commas."""
            response = model.generate_content(keyword_prompt)
            # Extract and return the first response
            return response.text if response and response.text else "No recommendation generated."
        except Exception as e:
            return f"An error occurred: {e}"

    def _extract_table_ddls(self, reranked_results: pd.DataFrame) -> List[str]:
        """Extract unique table DDLs from reranked results."""
        try:
            with open(self.config.metadata_path, 'r') as f:
                data = json.load(f)

            tables_set = reranked_results['tables'].unique()
            table_ddl = [
                self.create_ddl_from_json(data, table)
                for table in {
                    element for item in tables_set
                    for element in (item if isinstance(item, tuple) else (item,))
                }
            ]

            return table_ddl
        except Exception as e:
            self.logger.error(f"DDL extraction failed: {e}")
            return []

    def execute_workflow(
        self,
        query: str,
        metadata_path: Optional[str] = None,
        meta_weight: float = 1.4,
        query_weight: float = 0.9
    ) -> Optional[str]:
        """
        Execute complete RAG workflow with enhanced error handling.

        Args:
            query (str): User's natural language query
            metadata_path (str, optional): Override default metadata path
            meta_weight (float): Weight for metadata relevance
            query_weight (float): Weight for historical query relevance

        Returns:
            Optional[str]: Generated SQL recommendation
        """
        try:
            # Process query variations
            keywords = (self.get_input_variations(query)
                        if self.use_input_variations
                        else query)

            # Setup vector retrievals
            meta_data = VectorRetrieval(
                embedding_model=self.config.embedding_model,
                vector_db_path=self.config.meta_vector_path,
                data_type=VectorDataType.METADATA
            )

            query_data = VectorRetrieval(
                embedding_model=self.config.embedding_model,
                vector_db_path=self.config.queries_vector_path,
                data_type=VectorDataType.QUERYDATA
            )

            # Optional data ingestion
            if self.ingest:
                meta_data.ingest_data(
                    json_path=metadata_path or self.config.metadata_path
                )
                query_data.ingest_data(
                    json_path=self.config.queries_path
                )

            # Semantic search
            meta_results = meta_data.semantic_search(keywords)
            query_results = query_data.semantic_search(keywords)
            if len(meta_results) == 0 or len(query_results) == 0:
                self.logger.warning(f"No results found for query: {query}")
                return None
            # Merge and rerank results
            merged_df = self.merge_results(
                meta_results, query_results,
                meta_weight, query_weight
            )

            reranked_results = self.rerank_results(
                merged_df, query, meta_data.embedding_model
            )

            # Process table DDLs
            table_ddl = self._extract_table_ddls(reranked_results)

            # Generate SQL recommendation
            sql_recommendation = self.get_sql_recommendations(
                query, reranked_results, table_ddl
            )

            return sql_recommendation

        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            return None
