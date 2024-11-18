import json
import os
from dotenv import load_dotenv
import google.generativeai as genai
from RAG.MetadataRAG import MetadataRAG
from RAG.QuerydataRAG import QuerydataRAG


def get_sql_recommendations(prompt: str) -> str:
    """
    Use Google Generative AI to get SQL recommendations.

    Parameters:
        prompt (str): A natural language prompt describing the SQL query or recommendations you want.

    Returns:
        str: The recommended SQL query.
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


def get_rerank_recommendations(prompt: str) -> str:
    """
    Use Google Generative AI to get SQL recommendations.

    Parameters:
        prompt (str): A natural language prompt describing the SQL query or recommendations you want.

    Returns:
        str: The recommended SQL query.
    """
    try:
        # Call the model with the prompt
        model = genai.GenerativeModel(model_name='gemini-1.5-flash',
                                      system_instruction=["""You're an expert SQL domain expert. Respond with JSON only in the schema:
                            'ranked_results': [
                                {
                                "query": "SQL QUERY",
                                "score": SCORE,
                                "origin": "historical_query OR metadata",
                                "reason": "reason why this is important"
                                                          
                                }]"""]

                                      )
        response = model.generate_content(prompt)
        # Extract and return the first response
        return response.text if response and response.text else "No recommendation generated."
    except Exception as e:
        return f"An error occurred: {e}"


def create_ddl_from_json(table_data, tablename):
    """
    Create DDL statement from JSON table definition.

    Args:
        table_data (dict): Dictionary containing table name and columns

    Returns:
        str: DDL statement for creating the table
    """

    # Extract table name and columns

    # Start building the CREATE TABLE statement
    ddl = f"CREATE TABLE {tablename} (\n"

    # Process each column
    column_definitions = []
    for table in table_data["tables"]:
        if table["table_name"] == tablename:
            for column in table["columns"]:
                # Build column definition
                column_def = f"    {column['column_name']} {
                    column['column_type']}"

                # Add comment if it exists and is not null
                if column.get('column_remarks'):
                    column_def += f" COMMENT '{column['column_remarks']}'"

                column_definitions.append(column_def)

    # Join all column definitions with commas
    ddl += ",\n".join(column_definitions)

    # Close the CREATE TABLE statement
    ddl += "\n);"

    return ddl


def init_meta_rag(keywords: str):
    meta_rag = MetadataRAG()
    # # # Load metadata from JSON file
    metadata = meta_rag.load_json_metadata("data/input/fake_columns.json")
    # print(metadata)
    # # # Add metadata to the RAG system
    meta_rag.add_data(metadata)

    meta_rag.save_index("metadata_rag")
    # meta_rag.load_index("metadata_rag")
    for query, weights in keywords:
        print(f"\nQuery: {query}")
        if weights:
            print("Custom weights:", weights)
        print("-" * 50)
        tables = []
        results = meta_rag.search(query, k=15, custom_weights=weights)
        # print(results)
        scoring_columns = []
        for meta, score in results:
            print(f"Score: {score:.3f}")
            print(f"Table: {meta.table_name}")
            tables.append(meta.table_name)
            print(f"Column: {meta.column_name}")
            print(f"Type: {meta.column_type}")
            print(f"Description: {meta.column_remarks}")
            print("-" * 30)
            retrieval = {"column_name": meta.column_name,
                         "score": score, "table_name": meta.table_name}
            scoring_columns.append(retrieval)
        tables = set(tables)
        with open('data/input/fake_columns.json', 'r') as f:
            data = json.load(f)
            table_schema = []
            for table in tables:
                table_schema.append(create_ddl_from_json(data, table))
    return scoring_columns, table_schema


def init_query_rag(keywords: list):
    query_rag = QuerydataRAG()
    querydata = query_rag.load_json_querydata("data/input/fake_queries.json")
    query_rag.add_data(querydata)

    query_rag.save_index("querydata_rag")
    # query_rag.load_index("metadata_rag")
    for query, weights in keywords:
        print(f"\nQuery: {query}")
        if weights:
            print("Custom weights:", weights)
        print("-" * 50)
        tables = []
        results = query_rag.search(query, k=15, custom_weights=weights)
        # print(results)
        scoring_queries = []
        for query, score in results:
            print(f"Score: {score:.3f}")
            print(f"Query: {query.query}")
            print(f"User: {query.user}")
            print(f"Tables: {query.tables}")
            print("-" * 30)
            retrieval = {"query": query.query,
                         "score": score, "user": query.user, "tables": query.tables}
            scoring_queries.append(retrieval)
            tables.append(query.tables)

    result_set = set(item for sublist in tables for item in sublist)

    with open('data/input/fake_columns.json', 'r') as f:
        data = json.load(f)
        table_schema = []
        for table in result_set:
            table_schema.append(create_ddl_from_json(data, table))

    return scoring_queries, table_schema


def main():
    # Initialize RAG system
    load_dotenv()
    query = "shipping best carrier performance  top5"
    genai.configure(api_key=os.getenv("gemini_api_key"))
    # init_meta_rag(["address state user email reviews rating"])
    scoring_query, tables_query = init_query_rag([
        (query, {  # Emphasize column names
            'user': 0.0,
            'query': 0.9,
            'tables': 0.6
        }),
    ]
    )
    scoring_meta, tables_meta = init_meta_rag([
        (query, {  # Emphasize column names
            'name': 0.6,
            'type': 0.1,
            'remarks': 0.2,
            'context': 0.7
        }),
    ]
    )

    ensemble_prompt = f"""
    You're a senior business intelligence engineer.
    Your team is working on getting knowledge from a database. The user input the following keywords: '{query}'.
    Based on this keywords we used metadata retrieval and got the following items: "{scoring_meta}".
    Based on this keywords we also used historical queries and got the following items "{scoring_query}".
    And the matching table schemas: {tables_meta}, {tables_query}.
    Scoring from historical queries should be more important.
    Please rerank the found items and get the best 10 results order by score.
    """
    rerank = get_rerank_recommendations(ensemble_prompt)
    # table_schema = [obj for obj in metadata.get("metadata", []) if obj.get("table_name") == table_name]
    prompt = f"""

        # SQL Query Generation Template
        ## User Input Keywords
        {query}

        ## Reranked Context
        {rerank}

        ## Schema Context
        ```sql
        {tables_meta}
        {tables_query}
        ```

        ## Query Requirements Template
        Please provide the following information to generate an optimized SQL query:

        1. **Primary Objective**
        - What is the main metric or insight you want to calculate based on the keywords?
        - Example: "Find active users based on order frequency"

        2. **Ranked Context**
        - Use the given Reranked Context to construct the query
        - Example: Examine reason and score for each item

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
        2. Include appropriate JOINs based on foreign key relationships
        3. Add specific WHERE clauses for time-based filtering
        4. Use window functions for ranking when needed
        5. Include HAVING clauses for aggregate filters
        6. Specify ORDER BY for clear result presentation

    """
    print("-" * 50)
    print(prompt)
    print(get_sql_recommendations(prompt))


if __name__ == "__main__":
    main()
