import pandas as pd
import ast
import json
import re
import os


# Folder containing your result CSVs
input_folder = "./llm_outputs"
output_folder = "./evaluation_outputs"
os.makedirs(output_folder, exist_ok=True)

# Iterate over all CSV files in the input folder
for file_name in os.listdir(input_folder):
    if not file_name.endswith(".csv"):
        continue  # Skip non-CSV files
    try:
        print(f"Processing {file_name}...")
        # Load the experimental results
        results_df = pd.read_csv(f"{input_folder}/{file_name}")
        results_df = results_df[~results_df['topology_type'].eq('ring')]
        graphs_df = pd.read_csv("../DatasetGen/topology_with_node_pair.csv") 
        graphs_df = graphs_df[~graphs_df['topology_type'].eq('ring')]
        results_df[['source', 'target']] = results_df['node_pair'].apply(ast.literal_eval).apply(pd.Series)
        # Add extracted graph topology
        graphs_df['graph_dict'] = graphs_df['topology'].apply(lambda x: {int(k): v for k, v in ast.literal_eval(x).items()})
        
        # Merge on 'graph_id' to bring in topology_dimension (assumed to be in graphs_df)
        # Merge results with graph metadata
        merged_df = pd.merge(
            results_df,
            graphs_df[[
                'graph_id', 'graph_dict', 'topology_dimension',
                'num_nodes', 'num_edges',
                'avg_degree', 'max_degree', 'min_degree',
                'density', 'is_connected', 'num_components',
                "avg_weight", "std_weight", "var_weight", "max_link_weight", "min_link_weight", "weight_range", "weight_to_max_weight_ratio", "ground_truth_shortest_path","min_total_cost"
            ]],
            on='graph_id',
            how='left'
        )
        
        if "param_number" not in merged_df.columns:
            merged_df["param_number"] = "Unknown"

        if "quantisation_level" not in merged_df.columns:
            merged_df["quantisation_level"] = "Unknown"

        # Filter timeout rows
        timeout_df = merged_df[merged_df["token_count_completion"] == "timeout"]
        
        # Grouping columns
        group_cols = ["topology_type", "topology_dimension", "weight_distribution", "LLM", "param_number","quantisation_level"]
        
        # Calculate timeout rate
        grouped_total = merged_df.groupby(group_cols).size().reset_index(name='total_count')
        grouped_timeouts = timeout_df.groupby(group_cols).size().reset_index(name='timeout_count')
        
        # Compute timeout stats
        timeout_stats = pd.merge(grouped_total, grouped_timeouts, on=group_cols, how='left')
        timeout_stats['timeout_count'] = timeout_stats['timeout_count'].fillna(0).astype(int)
        timeout_stats['timeout_rate'] = timeout_stats['timeout_count'] / timeout_stats['total_count']
        
        # # Filter non-timeout rows for further processing
        # merged_df = merged_df[merged_df["token_count_completion"] != "timeout"]
        # Replace "timeout" with 120 in the token_count_completion column
        merged_df["token_count_completion"] = merged_df["token_count_completion"].replace("timeout", 120)

        # Replace "timeout" with 120 in the response_time_duration column as well
        merged_df["response_time_duration"] = merged_df["response_time_duration"].replace("timeout", 120)

        # Replace empty lists [] in predicted_path with [0]
        #merged_df["predicted_path"] = merged_df["predicted_path"].apply(lambda x: [0] if x == [] else x)

        
        
        # --------------------- Run Evaluation Checks ---------------------
        
        # Function to check if predicted_path is a valid Python list
        def is_valid_list(s):
            try:
                parsed = ast.literal_eval(s)
                return isinstance(parsed, list)
            except (ValueError, SyntaxError):
                return False
        
        
        
        # Function to validate edge connectivity
        def check_all_edges_exist(row):
            graph = row['graph_dict']
            try:
                path = ast.literal_eval(row['predicted_path'])
            except:
                return False
        
            if len(path) > 1:
                for i in range(len(path) - 1):
                    current = path[i]
                    nxt = path[i + 1]
                    neighbors = [n for n, _ in graph.get(current, [])]  # extract neighbor node IDs
                    if nxt not in neighbors:
                        return False
                return True
            return False
        
        # Add weight for each edge
        # Format the output as: [(node1, weight1), (node2, weight2), ..., nodeN]
        # Add (node, weight) tuples to the predicted path
        def add_weight_to_each_edge(row):
            topo = row['graph_dict']
            try:
                predicted_path = ast.literal_eval(row['predicted_path'])
            except:
                return []
        
            formatted_path = []
            weights = []
            for i in range(len(predicted_path) - 1):
                u, v = predicted_path[i], predicted_path[i + 1]
                neighbors = topo.get(u, [])
                for nbr, weight in neighbors:
                    if nbr == v:
                        formatted_path.append(u)
                        formatted_path.append(({weight}))
                        weights.append(weight)
                        break
            # Append final node without weight
            if predicted_path:
                formatted_path.append(predicted_path[-1])
            if weights:
                total_cost = sum(weights)
            else:
                total_cost = 0
        
            return formatted_path, total_cost
        
        merged_df.drop
        merged_df['correct_format'] = merged_df['predicted_path'].apply(is_valid_list)
        
        
        merged_df['source_target_match'] = merged_df.apply(
            lambda row: ast.literal_eval(row['predicted_path'])[0] == row['source'] and
                        ast.literal_eval(row['predicted_path'])[-1] == row['target'],
            axis=1
        )
        
        merged_df['repeated_nodes'] = merged_df['predicted_path'].apply(
            lambda x: len(ast.literal_eval(x)) != len(set(ast.literal_eval(x)))
        )
        
        merged_df['continuity'] = merged_df.apply(check_all_edges_exist, axis=1)
        
        merged_df[['predicted_path_with_weight', 'pred_min_total_cost']] = merged_df.apply(add_weight_to_each_edge, axis=1, result_type='expand')

        merged_df["cost_deviation"] = merged_df.apply(
            lambda row: float((row["pred_min_total_cost"] / row["min_total_cost"]) == 1)
            if row["min_total_cost"] != 0 else None,
            axis=1
        ).round(5)
    
        # merged_df["cost_deviation"] = merged_df.apply(
        #     lambda row: (row["pred_min_total_cost"] / row["min_total_cost"] - 1)
        #     if row["min_total_cost"] != 0 else None,
        #     axis=1
        # ).round(3)

        # print(merged_df['predicted_path_with_weight'])
        merged_df["successful_shortest_path"] = merged_df.apply(
            lambda row: (
                row['correct_format'] and
                row['source_target_match'] and
                not row['repeated_nodes'] and
                row['continuity'] and
                (1 if row['pred_min_total_cost'] == row['min_total_cost'] else 0)
            ),
            axis=1
        )
        
        # --------------------- Save Full Evaluation CSV ---------------------
        bool_columns = [
            'correct_format', 'source_target_match', 'repeated_nodes',
            'continuity', "successful_shortest_path"
        ]
        
        merged_df[bool_columns] = merged_df[bool_columns].astype(int)
        
        column_order = [
            "graph_id", "topology_type", "weight_distribution", "topology_dimension", 'num_nodes', 'num_edges',
            "avg_degree", "max_degree", "min_degree", "density",
            "is_connected", "num_components", "avg_weight", "std_weight", "var_weight", "max_link_weight", "min_link_weight", "weight_range", "weight_to_max_weight_ratio","source", "target", 
            "ground_truth_shortest_path", "predicted_path_with_weight", "min_total_cost", "pred_min_total_cost", 
            "correct_format", "source_target_match", "repeated_nodes",
            "continuity", "cost_deviation", "successful_shortest_path", "LLM", "param_number","quantisation_level","token_count_prompt", "token_count_completion", "response_time_duration"
        ]
        
        merged_df = merged_df[column_order + ['graph_dict']]
        #merged_df.drop(columns=['graph_dict']).to_csv(f"{output_folder}/evaluation_{file_name}", index=False)

        # --------------------- Save Concise CSV ---------------------
        merged_df.drop(columns=['graph_dict']).to_csv(f"{output_folder}/concise_evaluation_{file_name}", index=False)
        
        # print(merged_df)
        
        # Clean the token_count_prompt column
        # merged_df['token_count_prompt'] = pd.to_numeric(merged_df['token_count_prompt'], errors='coerce')
        
        # (Optional) Print out rows that were converted to NaN for inspection
        for col in ['token_count_prompt', 'token_count_completion', 'response_time_duration']:
            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
            # if not invalid_rows.empty:
            #     print("Warning: Non-numeric values found in token_count_prompt. Here are the affected rows:")
            #     print(invalid_rows[['graph_id', 'token_count_prompt']])
        # --------------------- Aggregation Summary by Topology Dimension ---------------------
        # Step 1: Evaluate only on non-timeout rows already in merged_df
        eval_numeric_cols = [
            "num_nodes", "num_edges", "avg_degree", "density",
            "correct_format", "source_target_match", "repeated_nodes", "continuity", "cost_deviation",
            "successful_shortest_path", "token_count_prompt", "token_count_completion", "response_time_duration"
        ]
     
        # Group and aggregate only non-timeout rows
        aggregated_metrics_df = merged_df[group_cols + eval_numeric_cols].groupby(group_cols).mean().reset_index()
        
        # Step 2: Ensure group_cols are strings for safe merging
        for col in group_cols:
            timeout_stats[col] = timeout_stats[col].astype(str)
            aggregated_metrics_df[col] = aggregated_metrics_df[col].astype(str)
        
        # Step 3: Merge timeout stats (ALL groups) with evaluation metrics (NON-timeout groups)
        aggregated_df = pd.merge(timeout_stats, aggregated_metrics_df, on=group_cols, how='left')
        
        # Round all numeric fields
        aggregated_df['timeout_rate'] = aggregated_df['timeout_rate'].round(3)
        aggregated_df = aggregated_df.round(1)
        
        # Step 4: Rename columns
        aggregated_df.rename(columns={
            "num_nodes": "avg_node_count",
            "num_edges": "avg_edge_count",
            "avg_degree": "avg_degree_avg",
            "density": "density_avg",
            "correct_format": "correct_format_rate",
            "source_target_match": "source_target_match_rate",
            "repeated_nodes": "repeated_nodes_rate",
            "continuity": "continuity_rate",
            "cost_deviation":"cost_deviation_avg",
            "successful_shortest_path": "success_rate"
            # "topology_type": "topo"
        
        }, inplace=True)
        
        # --------------------- Save aggregated_evaluation_results to CSV ---------------------
        # aggregated_df.drop(columns=['token_count_completion']).to_csv(f"{output_folder}/aggregated_evaluation_{file_name}", index=False)
        aggregated_df.to_csv(f"{output_folder}/aggregated_evaluation_{file_name}", index=False)
    
    except Exception as e:
        print(f"Failed to process {file_name}: {e}")

