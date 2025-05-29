import json
import os
import jsonast

def loadcallnavi():
    """
    Load the call navigation data from a file.
    
    Returns:
        list: A list of dictionaries containing the call navigation data.
    """
    prompt_1='''I have a JSON template for an API request:\n'''
    prompt_2='''\n
    Here is the question, Could you give me the ONLY the JSON related to the question?
    Question:
    '''

    datalst = ['bank', 'shopping', 'logistics', 'aviation', 'hospital', 'gov', 'hr', 'hotel', 'insurance', 'telecommunications']
    rows = []
    for data in datalst:
        with open("./CallNavi/Questions/" + data + ".json", 'r') as file:
            questions = json.load(file)

        with open("./CallNavi/APIs/" + data + ".json", 'r') as file:
            API = json.load(file)["api_ports"]
    
        with open("./CallNavi/APISchema/" + data + ".json", 'r') as file:
            schema = json.load(file)
        # Create a list to hold the rows
        
        for i in range(len(questions)):
            row={}
            for q in questions[i]["ground_truth"]["API"]:
                para=[]
                sch = []
                for a in API:
                    if a["name"] == q:
                        para.append(a)
                for s in schema:
                    if s["name"] == q:
                        sch.append(s)
            row["API"] = para
            row["schema"] = sch
            row["question"] = questions[i]["question"]
            row["answer"] = questions[i]["ground_truth"]
            rows.append(row)
        
    return rows
# Example usage
def test_callnavi(generation, ground_truth):
    """
    Test the generated output against the ground truth.

    Args:
        generation (str): The generated output.
        ground_truth (str): The expected output.

    Returns:
        bool: True if the generation matches the ground truth, False otherwise.
    """

    return jsonast.compare_jsons(generation, ground_truth, ["$$$"])


if __name__ == "__main__":
    data = loadcallnavi()
    for entry in data[1:2]:
        print("Question:", entry["question"])
        print("API:", entry["API"])
        print("Schema:", entry["schema"])
        print("Answer:", entry["answer"])
        print("-" * 50)