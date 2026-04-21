import ast
import pandas as pd
import json
import openai
import re
import time
import signal
import tiktoken
from itertools import islice
import os

client = openai.OpenAI(api_key = "...")


# Load dataset
# Load dataset
test_cases = pd.read_csv("../../DatasetGen/topology_with_node_pair.csv")
test_cases["topology"] = test_cases["topology"].apply(ast.literal_eval)
test_cases["node_pair"] = test_cases["node_pair"].apply(ast.literal_eval)


# # Extract cost from GPT response
# def extract_path(text):
#     """
#     Extracts the node path from a string like 'Answer: [0, 1, 2]'
#     Returns [0] if any invalid characters are present in the list.
#     """
#     match = re.search(r'Answer:\s*\[(.*?)\]', text)
#     if match:
#         list_str = match.group(1)
#         try:
#             # Ensure only digits and commas/spaces are present
#             if not re.fullmatch(r'[\d,\s]*', list_str):
#                 return [0]
#             # Convert to list of ints
#             return list(map(int, list_str.split(',')))
#         except Exception:
#             return [0]
#     return [0]


def extract_path(text):
    # Try standard Python list format
    match = re.search(r"\[([\d,\s]+)\]", text)
    if match:
        return [int(x.strip()) for x in match.group(1).split(",") if x.strip().isdigit()]
    
    # Try "4 -> 5 -> 6" or "4 → 5 → 6"
    match = re.findall(r"\b\d+\b(?:\s*[-→>]+\s*\d+\b)+", text)
    if match:
        path_str = match[0]
        return [int(n) for n in re.split(r"[-→>\s]+", path_str) if n.isdigit()]
    
    # Try digit sequences like "Path: 1, 2, 3"
    match = re.findall(r"\b(?:\d+\s*,\s*)+\d+\b", text)
    if match:
        return [int(x.strip()) for x in match[0].split(",") if x.strip().isdigit()]
    
    return []

# Run GPT evaluation
results = []

# Prompt template
SYSTEM_PROMPT = (
    "You are a helpful assistant that solves shortest path routing problems in computer networks. "
    "You are given a network topology in adjacency list format with weight: "
    "{node1: [(node2, weight2), (node3, weight3)], node2: [(node1, weight1)]}. "
    "Source_node and destination_node are in this format [source_node, destination_node]. "
    "According to the given topology, find the shortest path without code or explanation. "
    "The result should be a list of nodes like Answer: [0,1,2]."
)

# Timeout mechanism
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Model generation timed out.")
signal.signal(signal.SIGALRM, timeout_handler)

target_cases = test_cases.iloc[[2680,2689]]

# Limit to subset while testing to avoid quota exhaustion
for i, (index, entry) in enumerate(target_cases.iterrows()):
    topology_str = str(entry["topology"])
    node_pair_str = str(entry["node_pair"])
    messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(entry["topology"])},
                {"role": "user", "content": json.dumps(entry["node_pair"])}
            ]
    try:

        start_time = time.time()

        response = client.chat.completions.create(
            model = "gpt-5-nano",
            messages = messages
        )


        end_time = time.time()
        

        # Count tokens

        output = response.choices[0].message.content
        #print("output: ", output)
        predicted_path = extract_path(output)
        token_count_prompt = response.usage.prompt_tokens
        token_count_completion = response.usage.completion_tokens

        results.append({
                    "graph_id": entry["graph_id"],
                    "topology_type": entry["topology_type"],
                    "weight_distribution": entry["weight_distribution"],
                    "node_pair": entry["node_pair"],
                    "predicted_path": predicted_path,
                    "LLM": "GPT-5-nano",
                    "param_number": "Unknown",
                    "quantisation_level": "Unknown",
                    "token_count_prompt": token_count_prompt,
                    "token_count_completion": token_count_completion,
                    "response_time_duration": end_time - start_time
                })

        time.sleep(1)  # polite delay to avoid rate limit

    except TimeoutException as te:
        print(f"[✗] Timeout on entry {i}: {te}")
        signal.alarm(0)
        results.append({
            "graph_id": entry["graph_id"],
            "topology_type": entry["topology_type"],
            "weight_distribution": entry["weight_distribution"],
            "node_pair": entry["node_pair"],
            "predicted_path": [],
            "LLM": "GPT-5-nano",
            "param_number": "Unknown",
            "quantisation_level": "Unknown",
            "token_count_prompt": "timeout",
            "token_count_completion": "timeout",
            "response_time_duration": "timeout"
        })
        continue

    except Exception as e:
        print(f"[✗] Error on entry {i}: {e}")
        continue

# Convert to DataFrame
df_results = pd.DataFrame(results)

# Save to CSV
output_csv_path = f"weighted_GPT-5-nano_results.csv"
df_results.to_csv(output_csv_path, index=False)

# Show first few results in terminal
print("Finished !")
