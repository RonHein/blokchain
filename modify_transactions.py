import json
import pandas as pd
import numpy as np
import sys
import os

# Configuration
BLOCKS_PER_INTERVAL = 50  # Transactions grouped per 50 blocks
PUMP_THRESHOLD = 2500  # Total ETH required to trigger a pump event
WHALE_THRESHOLD = 1000  # Individual transactions above this are whales
PUMP_INTERVALS = 3  # Consecutive intervals where transactions will increase

def load_jsonl(file_path):
    """Loads a JSONL file into a Pandas DataFrame."""
    data = []
    with open(file_path, "r") as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return pd.DataFrame(data)

def save_jsonl(df, output_path):
    """Saves a DataFrame to a JSONL file."""
    with open(output_path, "w") as file:
        for record in df.to_dict(orient="records"):
            file.write(json.dumps(record) + "\n")

def modify_transactions(file_paths, output_folder="modified_data"):
    # Load all JSONL files
    df_list = [load_jsonl(fp) for fp in file_paths]
    df = pd.concat(df_list, ignore_index=True)

    # Convert transaction values from hex or string if needed
    def safe_eth_conversion(value):
        if isinstance(value, str) and value.startswith("0x"):
            return int(value, 16) / 10**18
        elif isinstance(value, (int, float)):
            return value / 10**18
        return 0

    df["value"] = df["transaction"].apply(lambda tx: safe_eth_conversion(tx.get("value", "0")))
    df["block_number"] = df["block_number"].astype(int)
    df["block_interval"] = df["block_number"]
