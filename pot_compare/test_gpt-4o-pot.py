import pandas as pd
import ast
import time
import re
import json
import openai
import signal
import tiktoken
from itertools import islice
import os

client = openai.OpenAI(api_key = "...")

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


def count_tokens(messages, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    
    num_tokens = 0
    for message in messages:
        # Every message follows: <|start|>{role/name}\n{content}<|end|>
        num_tokens += 4  # base tokens for each message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
    num_tokens += 2  # every reply is primed with <|start|>assistant
    return num_tokens

# Timeout mechanism
class TimeoutException(Exception):
    pass

# Run GPT evaluation
results = []

import ast

import ast
from collections import deque

import ast
import random
import networkx as nx

def find_valid_paths(adj_list_data, source, target, max_paths=8):
    """Finds the top K shortest paths mathematically, then shuffles them for the LLM."""
    try:
        if isinstance(adj_list_data, dict):
            raw_graph = adj_list_data
        elif isinstance(adj_list_data, str):
            raw_graph = ast.literal_eval(adj_list_data)
        else:
            return "Path 1: []"
    except Exception as e:
        return f"Path 1: [] (Parse Error: {e})"

    # 1. Build a NetworkX Graph from your adjacency list
    G = nx.Graph() # Use nx.DiGraph() if your edges are strictly directional
    for u, edges in raw_graph.items():
        for v, weight in edges:
            # Ensure nodes are strings to prevent type mismatches
            G.add_edge(str(u), str(v), weight=float(weight))

    norm_source, norm_target = str(source), str(target)
    valid_paths = []

    try:
        # 2. Use Yen's K-Shortest Paths algorithm (built into NetworkX)
        # This mathematically GUARANTEES the absolute shortest path is found first
        path_generator = nx.shortest_simple_paths(G, norm_source, norm_target, weight='weight')
        
        for i, path in enumerate(path_generator):
            if i >= max_paths:
                break
            valid_paths.append(path)
            
    except nx.NetworkXNoPath:
        return "Path 1: []"
    except nx.NodeNotFound:
        return "Path 1: []"

    # 3. CRITICAL: Shuffle the paths!
    # If the optimal path is always "Path 1", the LLM might just lazily guess "Path 1".
    # Shuffling forces the LLM to actually do the math to find which one is best.
    random.shuffle(valid_paths)

    # 4. Format output for the prompt
    formatted_output = ""
    for i, path in enumerate(valid_paths):
        clean_path = [int(n) if str(n).isdigit() else n for n in path]
        formatted_output += f"Path {i+1}: {clean_path}\n"
        
    return formatted_output if valid_paths else "Path 1: []"

# 2. Load Dataset (FILTER FOR FAT-TREE / UNIFORM to match your table)
test_cases = pd.read_csv("../../DatasetGen/topology_with_node_pair.csv")
test_cases["topology"] = test_cases["topology"].apply(ast.literal_eval)
test_cases[['source', 'target']]  = test_cases['node_pair'].apply(ast.literal_eval).apply(pd.Series)
# 3. The True PoT Prompt (Matches PoT Stage 3: Reasoning)
SYSTEM_PROMPT_TRUE_POT = (
    "You are an expert network routing assistant evaluating paths. "
    "Here is the network topology: {topology}\n\n"
    "Your task is to find the optimal shortest path from {source} to {target}.\n\n"
    "Reasoning chains (Valid Paths extracted via DFS):\n{valid_paths}\n\n"
    "Evaluate the total cumulative weight of each provided path step-by-step. "
    "Compare the totals and conclude with exactly the optimal path formatted as Answer: [0, 1, 2]."
)

target_cases = pd.concat([test_cases.iloc[590:640], test_cases.iloc[2510:2550]])
# 4. Execution Loop
results = []
for i, (index, entry) in enumerate(target_cases[86:87].iterrows()):
    # node_pair = str(entry["node_pair"])
    print(entry['source'], entry['target'])
    
    # Run the DFS to get paths
    paths_for_prompt = find_valid_paths(entry['topology'], entry['source'], entry['target'])
    
    # Inject into prompt
    user_prompt = SYSTEM_PROMPT_TRUE_POT.format(
        topology=entry['topology'], 
        source=entry['source'],
        target=entry['target'],
        valid_paths=paths_for_prompt
    )
   
    print(user_prompt)
    messages = [
                {"role": "user", "content": user_prompt}
    ]
    
    start_time = time.time()
    
    try:

        start_time = time.time()

        response = client.chat.completions.create(
            model = "gpt-4o",
            messages = messages,
            temperature = 0
        )


        end_time = time.time()
        

        # Count tokens

        output = response.choices[0].message.content
        print("output: ", output)
        predicted_path = extract_path(output)
        token_count_prompt = response.usage.prompt_tokens
        token_count_completion = response.usage.completion_tokens

        results.append({
                    "graph_id": entry["graph_id"],
                    "topology_type": entry["topology_type"],
                    "weight_distribution": entry["weight_distribution"],
                    "node_pair": entry["node_pair"],
                    "predicted_path": predicted_path,
                    "LLM": "GPT-4o",
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
            "LLM": "GPT-4o",
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
output_csv_path = f"weighted_GPT-4o_pot_results.csv"
df_results.to_csv(output_csv_path, index=False)

# Show first few results in terminal
print("Finished !")

