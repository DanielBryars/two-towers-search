from fastapi import FastAPI, Request
from pydantic import BaseModel
import gensim.downloader as api
import torch
import numpy as np
import faiss
import pickle
from model import load_checkpoint, text_to_embedding
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()
app.mount("/images", StaticFiles(directory="images"), name="images")

# Global state
w2v_model = None
query_model = None
index = None
documents = None

class Query(BaseModel):
    text: str

#uvicorn 04_nn_fastapi:app --reload --host 0.0.0.0 --port 8000

@app.on_event("startup")
def load_assets():
    global w2v_model, query_model, index, documents

    word2VecModelName = "word2vec-google-news-300"
    print(f"Loading Word2Vec model '{word2VecModelName}' ...")
    w2v_model = api.load("word2vec-google-news-300")

    querymodel_filename = "checkpoints/ts.2025_04_25__11_28_01.epoch.5.twotower.pth"
    print(f"Loading query model '{querymodel_filename}'...")
    query_model, _ = load_checkpoint(querymodel_filename)

    documentEmbeddings_filepath = "documentEmbeddings.v2.pkl"
    print(f"Loading document embeddings '{documentEmbeddings_filepath}'...")
    with open(documentEmbeddings_filepath, "rb") as f:
        embeddings, documents = pickle.load(f)

    embeddings = embeddings.astype("float32")
    index = faiss.IndexFlatL2(128)
    index.add(embeddings) # type: ignore

    print("Assets loaded.")

@app.get("/")
def serve_index():
    return FileResponse("index.html")

@app.post("/search")
def search(query: Query):
    global w2v_model, query_model, index, documents

    # Embed query
    q_embed = text_to_embedding(query.text, w2v_model) # type: ignore
    with torch.no_grad():
        q_tensor = torch.from_numpy(q_embed).unsqueeze(0)
        q_embed = query_model(q_tensor).squeeze(0).numpy().reshape(1, -1).astype("float32") # type: ignore

    # Search
    distances, indices = index.search(q_embed, k=10) # type: ignore

    results = [
        {"document": documents[i], "distance": float(d)} # type: ignore
        for i, d in zip(indices[0], distances[0])
    ]
    return {"results": results}
