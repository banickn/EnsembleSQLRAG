from RAG.SqlRAG import SqlRAG


def main():

    recommender = SqlRAG(use_input_variations=False)
    print(recommender.execute_workflow(
        "give me all warehouses with low stock inventory items"))


if __name__ == "__main__":
    main()
