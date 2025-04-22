import pickle
import os
import gensim.downloader as api
from gensim.models import KeyedVectors 
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.typing import NDArray

from huggingface_hub import HfApi

hfapi = HfApi(token=os.getenv("HF_TOKEN"))

print ("loading word2vec model...")
model = api.load("word2vec-google-news-300")
print ("word2vec model loaded")

def text_to_embedding(text: str, model:KeyedVectors) -> NDArray[np.float32]:
    words = text.lower().split()
    vectors = [model[word] for word in words if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size) # type: ignore

def compute_embeddings(filename):
  df = pd.read_parquet(filename)
  query_vectors = {}
  passage_vectors = {}

  for _, row in tqdm(df.iterrows(), total=len(df)):
      qid = row['query_id']
      query = row['query']
      positive_matches = row['passages']['passage_text']
      query_vectors[qid] = text_to_embedding(query, model)     # type: ignore
      passage_vectors[qid] = np.array([text_to_embedding(passage, model) for passage in positive_matches]) # type: ignore

  return query_vectors, passage_vectors

for filename in ["test-00000-of-00001.parquet", "train-00000-of-00001.parquet","validation-00000-of-00001.parquet"]:
  print(f"computing embeddings for '{filename}'")
  query_vectors, passage_vectors = compute_embeddings(filename)

  pickle_filename = f"{filename}.embeddings.pkl" 
  print(f"saving embeddings to '{pickle_filename}'")
  with open(pickle_filename, "wb") as f:
    pickle.dump((query_vectors, passage_vectors), f)

  print(f"uploading '{pickle_filename}' to huggingface")
  hfapi.upload_file(
    path_or_fileobj=pickle_filename,
    path_in_repo=pickle_filename,
    repo_id="danbhf/two-towers",
    repo_type="dataset",  # or "model" depending on your use
  )

# Load
#with open("vectors.pkl", "rb") as f:
#    query_vectors, passage_vectors = pickle.load(f)