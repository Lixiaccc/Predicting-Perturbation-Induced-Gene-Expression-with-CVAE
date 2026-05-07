#!/usr/bin/env python3
"""
01_generate_genept.py

Generate GenePT embeddings for the 9 KO genes used in this project.
Follows the same recipe as epifoundatoin_v2_gene/generate_genept_embeddings.py:
  - description: "The gene {X} is a human protein-coding gene."
  - model:       text-embedding-3-large (1536-D)
  - L2 normalize the resulting vector

Reads the OpenAI key from environment variable OPENAI_API_KEY.
Writes one CSV at HIGH_DIM_final/processed/projectors/genept_embeddings.csv:
  index = gene name, columns = 1536-D embedding components.

Usage:
  export OPENAI_API_KEY=sk-...
  python 01_generate_genept.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KO_GENES = [
    "ACTL6A", "DMAP1", "EP400", "EZH2",
    "SMARCA4", "SMARCB1", "SMARCE1", "SUZ12", "YY1",
]

OUT_PATH = Path(
    "/insomnia001/depts/houlab/users/lc3716/HIGH_DIM_final/processed/projectors/genept_embeddings.csv"
)


def gene_description(gene_name: str) -> str:
    return f"The gene {gene_name} is a human protein-coding gene."


def get_embedding(client, text: str, model: str = "text-embedding-3-large") -> np.ndarray:
    resp = client.embeddings.create(input=[text], model=model)
    return np.array(resp.data[0].embedding, dtype=np.float32)


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY env var not set. Run: export OPENAI_API_KEY=sk-...")

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    for gene in KO_GENES:
        desc = gene_description(gene)
        for attempt in range(3):
            try:
                emb = get_embedding(client, desc)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                results[gene] = emb
                print(f"  {gene}: ok (norm={norm:.4f})")
                break
            except Exception as e:
                print(f"  {gene}: error attempt {attempt+1}: {e}")
                time.sleep(2)
        else:
            sys.exit(f"Failed to embed {gene} after 3 retries")

    df = pd.DataFrame(results).T
    df.index.name = "gene"
    df.to_csv(OUT_PATH)
    print(f"\nSaved {len(df)} embeddings (shape {df.shape}) to {OUT_PATH}")


if __name__ == "__main__":
    main()
