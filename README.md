# PROMPT: Probing Reasoning in LLMs Over Multi-Step Pathfinding Tasks

[![Paper](https://img.shields.io/badge/Paper-Expert%20Systems%20with%20Applications-blue)](#)
[![Data License](https://img.shields.io/badge/Data%20License-MIT-green.svg)](#)

This repository contains the complete dataset, raw inference logs, and prompt configurations for the paper **"PROMPT: Probing Reasoning in LLMs Over Multi-Step Pathfinding Tasks"** (submitted to *Expert Systems with Applications*). 

Our study evaluates the reasoning traces of 14 Large Language Models (LLMs) across a controlled shortest-path routing testbed, systematically varying graph topologies (Grid, Fat-Tree, Jellyfish, Erdős–Rényi, Barabási–Albert) and edge-weight distributions (fixed, uniform, discrete, lognormal).

## 📊 Data Transparency & Reproducibility

Because the use of rolling model aliases, default generation temperatures, and varying structural prompts precludes strict byte-for-byte reproducibility from the API endpoints alone, we have prioritised **total data transparency**. 

To ensure our findings are fully verifiable and extensible by the research community, the complete, raw logs of all structured inputs, exact prompt strings, and corresponding model-generated outputs used in our analysis have been made publicly available in this repository.

## ⚙️ Decoding Parameters & Inference Pipeline

To rigorously evaluate both open-source and proprietary models, we utilised specific decoding configurations. The parameters used to generate the data in this repository are outlined below:

* **Local Open-Source Models (10 Models):** Executed via the HuggingFace `transformers` stack using strict greedy decoding to ensure deterministic evaluation.
    * `do_sample = False`
    * `temperature = None`
* **Proprietary Models:**
    * **GPT-4o:** Queried via the OpenAI API with a constrained `temperature = 0`.
    * **GPT-5 Suite:** Queried via the API using the models' default generation parameters to capture their baseline reasoning behaviour.

## 📂 Repository Structure

```text
├── ablation_test/
│   ├── POT_compare/
│   │   ├── evaluation_outputs/
│   │   │   ├── aggregated_evaluation_weighted_GPT-5-mini_pot_results.csv
│   │   │   ├── aggregated_evaluation_weighted_GPT-5-mini_results.csv
│   │   │   ├── concise_evaluation_weighted_GPT-5-mini_pot_results.csv
│   │   │   └── concise_evaluation_weighted_GPT-5-mini_results.csv
│   │   ├── llm_outputs/
│   │   │   ├── weighted_GPT-5-mini_pot_results.csv
│   │   │   └── weighted_GPT-5-mini_results.csv
│   │   ├── evaluation.py
│   │   ├── final_aggreagation.py
│   │   └── test_gpt-4o-pot.py
│   ├── Prompting_Strategies/
│   │   ├── llm_outputs/
│   │   ├── evaluation.py
│   │   ├── final_aggreagation.py
│   │   ├── test_gpt-4o-cot-oneshot.py
│   │   ├── test_gpt-4o-cot.py
│   │   └── test_gpt-4o-oneshot.py
│   └── standard_zero_shot/
│       ├── Stage1_Zero-Shot_Answer_Generation/
│       │   ├── llm_ouputs/
│       │   ├── test_gpt-4o.py
│       │   ├── test_gpt-5-mini.py
│       │   ├── test_gpt-5-nano.py
│       │   └── test_gpt-5.py
│       └── Stage2_Retrospective_Reasoning_Extraction/
│           ├── Reasoning_process/
│           ├── test_gpt-4o_reasoning.py
│           ├── test_gpt-5-mini_reasoning.py
│           ├── test_gpt-5-nano_reasoning.py
│           └── test_gpt-5_reasoning.py
└── README.md
