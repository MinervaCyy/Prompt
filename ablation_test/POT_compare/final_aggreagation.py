import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Patch
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages

# === Configuration ===
input_folder = "./evaluation_outputs/"
output_file = "./result_analysis_out/final_aggregation"
os.makedirs(output_file, exist_ok=True)

# === Define Orders ===
weight_distribution = [ "fixed", "discrete", "uniform", "lognormal"]
unique_topos = ["grid", "fat_tree", "jellyfish", "erdos_renyi", "barabasi_albert"]

# === Define Global Averages ===
metrics = [
    "success_rate",
    "correct_format_rate",
    "source_target_match_rate",
    "repeated_nodes_rate",
    "continuity_rate",
    "cost_deviation_avg"
    
]

# === Load and Annotate All CSVs ===
aggregated_files = [
    f for f in os.listdir(input_folder)
    if f.startswith("aggregated_evaluation_") and f.endswith(".csv")
]

for file_name in aggregated_files:
    try:
        model_name = file_name.replace("aggregated_evaluation_", "").replace(".csv", "")
        df = pd.read_csv(os.path.join(input_folder, file_name))

        # # Normalize to 'LLM' if needed
        # cols_lower = {c.lower(): c for c in df.columns}
        # if 'llm' in cols_lower:
        #     df.rename(columns={cols_lower['llm']: 'LLM'}, inplace=True)
        # elif 'model' in cols_lower:
        #     df.rename(columns={cols_lower['model']: 'LLM'}, inplace=True)
        #     df['LLM'] = df['LLM'].fillna(model_name)
        # else:
        #     df['LLM'] = model_name

        #df_list.append(df)
        averages = df[metrics].mean().round(5)

        # === Compute per-topology ===
        # Per-topology means (treat each topology equally)
        topo_means = df.groupby("topology_type")["success_rate"].mean()
        
        n_topos = topo_means.size
        
        # Unweighted across topologies (each topology equally)
        avg_topos = topo_means.mean() if n_topos > 0 else np.nan
        sem_topos = (topo_means.std(ddof=1) / np.sqrt(n_topos)) if n_topos > 1 else np.nan
        
        # Weighted across topologies (weight each topology by its number of rows)
        # Equivalent to df["success_rate"].mean(), but expressed via topo_means for clarity
        counts = (
            df["topology_type"]
            .value_counts()
            .reindex(topo_means.index)
            .astype(float)
        )
        
        weighted_avg_topos = (
            float(np.average(topo_means, weights=counts)) if n_topos > 0 else np.nan
        )
        
        # Standard error for the weighted mean, treating each ROW as an observation
        total_n = int(counts.sum()) if n_topos > 0 else len(df)
        weighted_sem_topos = (
            df["success_rate"].std(ddof=1) / np.sqrt(total_n) if total_n > 1 else np.nan
        )
        
        # Per-weight-distribution means (treat each distribution equally)
        weight_means = df.groupby("weight_distribution")["success_rate"].mean()
        #print(weight_means)
        n_weights = weight_means.size
        avg_weights = weight_means.mean() if n_weights > 0 else np.nan
        sem_weights = (weight_means.std(ddof=1) / np.sqrt(n_weights)) if n_weights > 1 else np.nan

        # Round for neat printing
        weighted_avg_topos = round(float(weighted_avg_topos), 5) if not np.isnan(weighted_avg_topos) else np.nan
        weighted_sem_topos = round(float(weighted_sem_topos), 5) if not np.isnan(weighted_sem_topos) else np.nan
        avg_weights = round(float(avg_weights), 5) if not np.isnan(avg_weights) else np.nan
        sem_weights = round(float(sem_weights), 5) if not np.isnan(sem_weights) else np.nan


        # # === Compute Robustness (SEM of success_rate) ===
        # # Across topologies
        # weighted_sem_topos = (
        #     df.groupby("topology_type")["success_rate"].mean().std(ddof=1)
        #     / np.sqrt(len(unique_topos))
        # )
        # weighted_sem_topos = round(weighted_sem_topos, 5)

        # # Across weight distributions
        # sem_weights = (
        #     df.groupby("weight_distribution")["success_rate"].mean().std(ddof=1)
        #     / np.sqrt(len(weight_distribution))
        # )
        # sem_weights = round(sem_weights, 5)

        results = pd.DataFrame(
            {
                "metric": list(averages.index)
                + [
                    "avg_success_rate_topology",
                    "sem_success_rate_topology",
                    "avg_success_rate_weight_distribution",
                    "sem_success_rate_weight_distribution",
                ],
                "value": list(averages.values)
                + [weighted_avg_topos, weighted_sem_topos, avg_weights, sem_weights],
            }
        )
        

        # # === Save Results ===
        # results = pd.DataFrame(
        #     {
        #         "metric": list(averages.index)
        #         + ["sem_success_rate_topology", "sem_success_rate_weight_distribution"],
        #         "value": list(averages.values) + [sem_topos, sem_weights],
        #     }
        # )

        #results.to_csv(output_file, index=False)

        print(f"{model_name}")
        print(results)
        print("--------------------------------------------------------")

    except Exception as e:
        print(f"Error reading {file_name}: {e}")

# if not df_list:
#     raise RuntimeError("No CSVs loaded. Check input_folder or filenames.")

# all_data = pd.concat(df_list, ignore_index=True)

# # If the caller forced 'model' earlier, ensure 'LLM' exists
# if 'LLM' not in all_data.columns and 'model' in all_data.columns:
#     all_data.rename(columns={'model': 'LLM'}, inplace=True)


# unique_llms = [
#     "GPT-5",
#     "GPT-5-mini",
#     "GPT-5-nano",
#     "GPT-4o",
#     "Llama-3.1-70B-Instruct",
#     "Llama-3.1-70B",
#     "DeepSeek-R1-Distill-Llama-70B"
#     #"Mistral-7B-Instruct-v0.3"
# ]

# for file_name in aggregated_files:
#     df_list = []
#     try:
#         model_name = file_name.replace("aggregated_evaluation_", "").replace(".csv", "")
#         df = pd.read_csv(os.path.join(input_folder, file_name))
#         df['model'] = model_name
#         df_list.append(df)


#     except Exception as e:
#         print(f"Error reading {file_name}: {e}")

# all_data = pd.concat(df_list, ignore_index=True)

# # === Categorize Columns for Ordered Plotting ===
# all_data['topology_type'] = pd.Categorical(all_data['topology_type'], categories=unique_topos, ordered=True)
# all_data['weight_distribution'] = pd.Categorical(all_data['weight_distribution'], categories=weight_distribution, ordered=True)

