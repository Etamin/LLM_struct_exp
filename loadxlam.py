from datasets import load_dataset
import json

def load_xlam_dataset(split="train", max_samples=None):
    """
    Load and process the xLAM function-calling dataset.

    Args:
        split (str): Dataset split to load ('train', 'test', etc.).
        max_samples (int, optional): Maximum number of samples to load.

    Returns:
        List[Dict]: A list of dictionaries containing 'query', 'tools', and 'answers'.
    """
    # Load the dataset
    dataset = load_dataset("Salesforce/xlam-function-calling-60k", split=split)

    # Optionally limit the number of samples
    if max_samples:
        dataset = dataset.select(range(max_samples))

    processed_data = []
    for item in dataset:
        try:
            # Parse the JSON strings
            tools = json.loads(item["tools"])
            answers = json.loads(item["answers"])
            query = item["query"]

            processed_data.append({
                "query": query,
                "tools": tools,
                "answers": answers
            })
        except json.JSONDecodeError as e:
            # Handle JSON parsing errors if any
            print(f"Error parsing JSON for item with query: {item['query']}")
            continue

    return processed_data


# Example usage
# data = load_xlam_dataset(max_samples=100)
# for entry in data[:5]:
#     print("Query:", entry["query"])
#     print("Tools:", entry["tools"])
#     print("Answers:", entry["answers"])
#     print("-" * 50)
