#!/usr/bin/env python3
"""
EXPERIMENTAL - NOT CURRENTLY USED
Build / update data/utils/player_dob.csv from Wikidata SPARQL.

This script is experimental and not part of the active pipeline. DOB data is currently populated via manual/LLM workflow.

Usage:
  python build_player_dob.py --season 25-26 --quotazioni data/raw/25-26/Quotazioni_Fantacalcio_Stagione_2025_26.xlsx

The script reads Id and Name from Quotazioni, merges with existing DOB CSV,
queries Wikidata for missing players in batches, and writes back data/utils/player_dob.csv
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "FantaHelpML/1.0 (duilio@example.com)"

def load_quotazioni(path):
    df = pd.read_excel(path, header=1)
    # Adjust column names to your Quotazioni schema
    # Expected: Id, Nome, ...
    # Normalize
    df = df.rename(columns={c: c.strip() for c in df.columns})
    return df

def load_existing_dob(csv_path):
    if Path(csv_path).exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame(columns=["Id","Name","DOB","source"])

def wikidata_query_names(names):
    # Build VALUES list for batch query
    values = " ".join(f'"{n}"' for n in names)
    query = f"""
    SELECT ?player ?playerLabel ?dateOfBirth WHERE {{
      VALUES ?label {{ {values} }}
      ?player rdfs:label ?playerLabel .
      FILTER(LCASE(?playerLabel) = LCASE(?label))
      ?player wdt:P106 wd:Q937857 ;  # football player
              wdt:P569 ?dateOfBirth .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    headers = {"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"}
    resp = requests.get(WIKIDATA_ENDPOINT, params={"query": query, "format": "json"}, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    results = {}
    for b in data.get("results", {}).get("bindings", []):
        label = b["playerLabel"]["value"]
        dob = b["dateOfBirth"]["value"].split("T")[0]
        results[label] = dob
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", required=True)
    parser.add_argument("--quotazioni", required=True)
    args = parser.parse_args()

    quot = load_quotazioni(args.quotazioni)
    # Expect columns Id, Nome
    if "Nome" not in quot.columns:
        raise ValueError("Quotazioni must contain 'Nome' column")
    players = quot[["Id","Nome"]].drop_duplicates()
    players = players.rename(columns={"Nome":"Name"})

    dob_path = Path("data/utils/player_dob.csv")
    existing = load_existing_dob(dob_path)
    # Merge
    merged = players.merge(existing[["Id","Name","DOB","source"]], on=["Id","Name"], how="left", suffixes=("","_old"))

    missing = merged[merged["DOB"].isna()][["Id","Name"]]
    print(f"Missing DOB for {len(missing)} players")

    batch_size = 50
    new_rows = []
    for i in range(0, len(missing), batch_size):
        batch = missing.iloc[i:i+batch_size]["Name"].tolist()
        try:
            res = wikidata_query_names(batch)
            time.sleep(0.5)
        except Exception as e:
            print(f"Query error: {e}")
            res = {}
        for name in batch:
            dob = res.get(name)
            if dob:
                new_rows.append({"Name": name, "DOB": dob, "source":"wikidata"})
    # Write back
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        # Merge back
        updated = pd.concat([existing, new_df], ignore_index=True).drop_duplicates(subset=["Name","Id"], keep="first")
    else:
        updated = existing

    # Ensure Id column exists
    # For simplicity, keep Id from players list
    final_df = players.merge(updated, on=["Id","Name"], how="left")
    final_df = final_df[["Id","Name","DOB","source"]].drop_duplicates()
    Path("data/utils").mkdir(parents=True, exist_ok=True)
    final_df.to_csv(dob_path, index=False)
    print(f"Wrote {len(final_df)} rows to {dob_path}")

if __name__ == "__main__":
    main()
