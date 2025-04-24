from fastapi import FastAPI, Request
from pydantic import BaseModel
import gensim.downloader as api
import torch
import numpy as np
import faiss
import pickle
from model import load_checkpoint, text_to_embedding

app = FastAPI()

# Global state
w2v_model = None
query_model = None
index = None
documents = None


class Query(BaseModel):
    text: str


@app.on_event("startup")
def load_assets():
    global w2v_model, query_model, index, documents

    print("Loading Word2Vec model...")
    w2v_model = api.load("word2vec-google-news-300")

    print("Loading query model...")
    query_model, _ = load_checkpoint("2025_04_23__12_47_41.11.twotower.pth")

    print("Loading document embeddings...")
    with open("documentEmbeddings.pkl", "rb") as f:
        embeddings, documents = pickle.load(f)

    embeddings = embeddings.astype("float32")
    index = faiss.IndexFlatL2(128)
    index.add(embeddings)

    print("Assets loaded.")


@app.post("/search")
def search(query: Query):
    global w2v_model, query_model, index, documents

    # Embed query
    q_embed = text_to_embedding(query.text, w2v_model)
    with torch.no_grad():
        q_tensor = torch.from_numpy(q_embed).unsqueeze(0)
        q_embed = query_model(q_tensor).squeeze(0).numpy().reshape(1, -1).astype("float32")

    # Search
    distances, indices = index.search(q_embed, k=10)

    results = [
        {"document": documents[i], "distance": float(d)}
        for i, d in zip(indices[0], distances[0])
    ]
    return {"results": results}
