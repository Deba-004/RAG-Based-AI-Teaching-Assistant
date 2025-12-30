import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import time

def merged_func(q):
    def create_embeddings(text_list):
        res = requests.post("http://localhost:11434/api/embed", json={
            "model": "bge-m3",
            "input": text_list
        })

        embedding = res.json()["embeddings"]
        return embedding
    
    def inference(prompt):
        res = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        })

        response = res.json()
        return response
    
    df = joblib.load("embeddings.joblib")

    t0 = time.time()
    query = q
    emb_query = create_embeddings([query])[0]
    embed_time = time.time() - t0

    t1 = time.time()
    similarities = cosine_similarity(np.vstack(df["embedding"]), [emb_query]).flatten()
    top_results = 6
    max_idx = np.argsort(similarities)[::-1][:top_results]
    new_df = df.loc[max_idx]
    retreival_time = time.time() - t1

    prompt = f'''Here are some video subtitle chunks containing video number, title, start time in seconds, end time in seconds and text content of that time chunk:
    {new_df[["number", "title", "start", "end", "text"]].to_json(orient="records")}
    ------------------------------------------------------------------------------------------------------------------------------------
    "{query}"
    User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course. And convert the seconds into minutes. You should also provide a summy of the topic.'''

    t2 = time.time()
    response = inference(prompt)["response"]
    llm_time = time.time() - t2

    print({
        "embed_time": round(embed_time, 4),
        "retreival_time": round(retreival_time, 4),
        "llm_time": round(llm_time, 4),
        "tottal_time": round(embed_time + retreival_time + llm_time, 4)
    })

    return response