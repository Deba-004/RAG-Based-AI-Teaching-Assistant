import requests
import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

def create_embeddings(text_list):
    res = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = res.json()["embeddings"]
    return embedding

jsons = os.listdir("jsons")
chunk_id = 0
my_dicts = []

for json_file in jsons:
    with open(f"merged_jsons/{json_file}") as f:
        data = json.load(f)
    print(f"Creating embeddings for {json_file}...")
    text_list = [c["text"] for c in data["chunks"]]
    embeddings = create_embeddings(text_list)
    for i, chunk in enumerate(data["chunks"]):
        chunk["chunk_id"] = chunk_id
        chunk_id += 1
        chunk["embedding"] = embeddings[i]
        my_dicts.append(chunk)
    print(f"Processed {json_file}")

df = pd.DataFrame.from_records(my_dicts)
joblib.dump(df, "embeddings.joblib")