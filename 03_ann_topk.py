import os
import sys
import pickle
import numpy as np
import torch
import faiss
import gensim.downloader as api
from huggingface_hub import HfApi
from model import load_checkpoint, text_to_embedding

if __name__ == "__main__":

   hfapi = HfApi(token=os.getenv("HF_TOKEN"))

   print("loading word2vec model...")
   w2v_model = api.load("word2vec-google-news-300")
   print("word2vec model loaded")

   print("loading query model...")
   queryModel, _ = load_checkpoint("2025_04_23__12_47_41.11.twotower.pth")
   print("query model loaded")

   print("loading document embeddings...")
   with open("documentEmbeddings.pkl", "rb") as f:
      embeddings, documents = pickle.load(f)
   print("document embeddings loaded")

   print("building index...")
   embeddings = embeddings.astype('float32')
   index = faiss.IndexFlatL2(128)
   index.add(embeddings)
   print("index built")

   while True:
      query = input("🔍 Query: ").strip()
      if query.lower() in {"exit", "quit"}:
         break

      print("")
      print("")
      print("")
    
      print(f"{query}")

      print("calculating query embedding...")
      query_embedding = text_to_embedding(query, w2v_model)  # type: ignore
      with torch.no_grad():
         query_tensor = torch.from_numpy(query_embedding).unsqueeze(0)  # (1, 300)
         query_embedding = queryModel(query_tensor).squeeze(0).numpy()
      print("query embedding calculated")

      query_embedding = query_embedding.reshape(1, -1)  # (1, 128)
      distances, indices = index.search(query_embedding, k=10)

      for i, (doc_idx, dist) in enumerate(zip(indices[0], distances[0])):
         print(f"{i+1:2d}. Distance: {dist:.4f}")
         print(f"{i}: --------------------------------")
         print(f"{documents[doc_idx]}")
         print("--------------------------------")