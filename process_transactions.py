#!/usr/bin/env python3

import json
import pandas as pd
from sklearn.ensemble import IsolationForest

def flatten_transaction_record(record):
    """
    Given a single transaction record from the JSONL (which typically has:
       - block_number
       - block_timestamp
       - transaction (dict)
       - receipt (dict)
    flatten it into a single dict of top-level fields + transaction fields + receipt fields.
    Returns (flattened_dict, logs_list).
    """
    flattened = {}
    logs_list = []

    # -------------------------------------------------------------------------
    # 1) Extract top-level fields
    # -------------------------------------------------------------------------
    flattened["block_number"] = record.get("block_number")
    flattened["block_timestamp"] = record.get("block_timestamp")

    # -------------------------------------------------------------------------
    # 2) Flatten 'transaction' fields (prefix them with 'tx_' to avoid collisions)
    # -------------------------------------------------------------------------
    tx = record.get("transaction", {})
    flattened["tx_hash"] = tx.get("hash")
    flattened["tx_from"] = tx.get("from")
    flattened["tx_to"]   = tx.get("to")
    flattened["tx_nonce"] = tx.get("nonce")
    flattened["tx_value"] = tx.get("value")
    flattened["tx_gas"]   = tx.get("gas")
    flattened["tx_gasPrice"] = tx.get("gasPrice")
    flattened["tx_input"]    = tx.get("input")
    flattened["tx_chainId"]  = tx.get("chainId")
    # ... add any other transaction fields you want

    # -------------------------------------------------------------------------
    # 3) Flatten 'receipt' fields (prefix them with 'rcpt_')
    # -------------------------------------------------------------------------
    rcpt = record.get("receipt", {})
    flattened["rcpt_status"] = rcpt.get("status")
    flattened["rcpt_gasUsed"] = rcpt.get("gasUsed")
    flattened["rcpt_contractAddress"] = rcpt.get("contractAddress")
    # ... add any other receipt fields you want

    # -------------------------------------------------------------------------
    # 4) Extract logs from the receipt
    # -------------------------------------------------------------------------
    logs = rcpt.get("logs", [])
    for log in logs:
        # Each log might have address, data, topics, etc.
        # We'll store them in a structure that references the transaction hash
        # (so we can link logs to transactions in another DataFrame).
        log_entry = {
            "tx_hash": tx.get("hash"),
            "block_number": record.get("block_number"),
            "logIndex": log.get("logIndex"),
            "address": log.get("address"),
            "data": log.get("data"),
            "removed": log.get("removed"),
            "topics": log.get("topics"),
        }
        logs_list.append(log_entry)

    return flattened, logs_list


def load_transactions_and_logs(jsonl_path):
    """
    Reads the .jsonl file line by line, flattens each record into:
      1) transaction-level data (transactions_df)
      2) logs-level data (logs_df)
    Returns two Pandas DataFrames.
    """
    all_tx = []
    all_logs = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            tx_flat, logs_list = flatten_transaction_record(record)

            all_tx.append(tx_flat)
            all_logs.extend(logs_list)

    transactions_df = pd.DataFrame(all_tx)
    logs_df = pd.DataFrame(all_logs)

    return transactions_df, logs_df


if __name__ == "__main__":
    # 1) Load the flattened data (in memory)
    jsonl_file = "transaction_data.jsonl"
    transactions_df, logs_df = load_transactions_and_logs(jsonl_file)

    # 2) Print quick previews
    print("Transactions DF:")
    print(transactions_df.head(5))
    print("\nLogs DF:")
    print(logs_df.head(5))

    print(f"\nNumber of transactions: {len(transactions_df)}")
    print(f"Number of logs: {len(logs_df)}")

    # 3) Example grouping: how many transactions each sender has
    tx_count_by_sender = transactions_df.groupby("tx_from")["tx_hash"].count()
    tx_count_by_sender.sort_values(ascending=False, inplace=True)
    print(f"\nTop 10 Senders:\n{tx_count_by_sender.head(10)}")

    # 4) Simple anomaly detection with IsolationForest
    #    We'll pick a few numeric fields
    features = ["rcpt_gasUsed", "tx_value", "tx_gasPrice"]
    # Fill missing numeric fields with 0
    X = transactions_df[features].fillna(0)

    # Create and fit the model
    iso = IsolationForest(contamination=0.01)  # ~1% anomalies
    iso.fit(X)

    # Store scores in the DataFrame
    # decision_function => higher = more normal, lower = more anomalous
    transactions_df["anomaly_score"] = iso.decision_function(X)
    # predict => -1 = anomaly, 1 = normal
    transactions_df["is_anomaly"] = iso.predict(X)

    # 5) Inspect suspicious transactions
    anomalies = transactions_df[transactions_df["is_anomaly"] == -1]
    print("\nAnomalies:")
    print(anomalies.head(20))

