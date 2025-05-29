from datasets import load_dataset
import sqlite3

# Adjust the import path to wherever your loader function is defined


def execute_query_on_schema(schema_sql: str, query_sql: str, query_gt) -> bool:
    """
    Create an in-memory SQLite database from the given schema SQL,
    execute the provided query SQL, and return True if successful, False otherwise.
    """
    conn = sqlite3.connect(':memory:')
    # print("Executing schema SQL:", schema_sql)
    # print("Executing query SQL:", query_sql)
    try:
        query_sql = query_sql.replace("```sql", "").replace("```", "").replace("\n"," ").strip()

        # Create tables and populate schema
        conn.executescript(schema_sql)
        gt= conn.execute(query_gt).fetchall()
        # Execute the query under test
        result=conn.execute(query_sql).fetchall()
        print("Query result:", result)
    except sqlite3.Error as e:
        return {"match":False, "error": "Query execution failed with error: " + str(e)}
    except Exception as e:
        return {"match":False, "error": "An unexpected error occurred: " + str(e)}
    finally:
        conn.close()
    # If we reach here, the query was executed successfully
    if result == gt:
        return {"match": True, "result": result}
    else:
        return {"match": False, "result": result, "gt": gt, "error": "Query result does not match ground truth."}




def load_cm_spider_dataset_old(split="train", max_samples=None):
    """
    Load and process the CM/spider dataset from Hugging Face.

    Args:
        split (str): Dataset split to load ('train' or 'test').
        max_samples (int, optional): Maximum number of samples to load.

    Returns:
        List[Dict]: A list of dictionaries containing 'question', 'query', 'db_id', and 'schema'.
    """
    # Load the dataset
    dataset = load_dataset("CM/spider", split=split)

    # Optionally limit the number of samples
    if max_samples:
        dataset = dataset.select(range(max_samples))

    # Process and return the data
    processed_data = []
    for item in dataset:
        processed_data.append({
            "question": item["question"],
            "query": item["query"],
            "db_id": item["db_id"],
            "schema": item["schema"]
        })

    return processed_data

def load_cm_spider_dataset(split: str = "test", max_samples: int = None) -> list[dict]:
    """
    Load and process the CM/spider dataset, extracting full CREATE statement blocks
    (which may span multiple lines) ending with a semicolon.

    Args:
        split (str): Dataset split to load ('train' or 'test').
        max_samples (int, optional): Maximum number of samples to load.

    Returns:
        List[Dict]: A list of dicts with 'question', 'query', 'db_id', and the
                    concatenated CREATE statement blocks as 'schema'.
    """
    # 1. Load the dataset
    dataset = load_dataset("CM/spider", split=split)

    # 2. Optionally limit the number of samples
    if max_samples is not None:
        dataset = dataset.select(range(max_samples))

    # 3. Define a function to extract CREATE blocks
    def _extract_create_blocks(example):
        raw_schema = example.get("schema", "")
        lines = raw_schema if isinstance(raw_schema, list) else raw_schema.splitlines()

        blocks = []
        current_block = []
        for ln in lines:
            stripped = ln.strip()
            # Start of a CREATE block
            if not current_block and stripped.upper().startswith("CREATE"):  
                current_block.append(ln)
                # If it ends and semicolon on same line
                if stripped.endswith(";"):
                    blocks.append("\n".join(current_block))
                    current_block = []
            # Continuation of block
            elif current_block:
                current_block.append(ln)
                if stripped.endswith(";"):
                    blocks.append("\n".join(current_block))
                    current_block = []
        # Join all found blocks
        filtered_schema = "\n\n".join(blocks)

        return {
            "question": example["question"],
            "query": example["query"],
            "db_id": example["db_id"],
            "create": filtered_schema,
            "schema": raw_schema
        }

    # 4. Map over the dataset for efficient batch processing
    processed = dataset.map(
        _extract_create_blocks,
        remove_columns=[c for c in dataset.column_names if c not in ["question", "query", "db_id", "schema"]]
    )

    # 5. Return as list of dicts
    return processed.to_list()

# Example usage
# if __name__ == "__main__":
#     data = load_cm_spider_dataset(split="train", max_samples=10)
#     for entry in data:
#         print("Question:", entry["question"])
#         print("Query:", entry["query"])
#         print("DB ID:", entry["db_id"])
#         print("create:", entry["create"])
#         print("Schema:", entry["schema"])
#         print("-" * 50)