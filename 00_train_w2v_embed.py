import pickle
import os
import gensim.downloader as api
from gensim.models import KeyedVectors 
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.typing import NDArray

from huggingface_hub import HfApi

'''
Creates the following structure:
{
    "query": np.ndarray of shape (N, D),
    "pos": np.ndarray of shape (N, D),
    "neg": np.ndarray of shape (N, D) #note this is just a randomised list of pos
}
'''

def text_to_embedding(text: str, model:KeyedVectors) -> NDArray[np.float32]:
    words = text.lower().split()
    vectors = [model[word] for word in words if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size) # type: ignore

def compute_embeddings(filename, model):
  df = pd.read_parquet(filename)
  query_embeddings = []
  positive_embeddings = []

  for _, row in tqdm(df.iterrows(), total=len(df)):
      query = row['query']
      positive_matches = row['passages']['passage_text']
      query_embedding = text_to_embedding(query, model)     # type: ignore
      
      for passage in positive_matches:          
        passage_embedding = text_to_embedding(passage, model) # type: ignore

        #pair of a query with a positive embedding
        query_embeddings.append(query_embedding)
        positive_embeddings.append(passage_embedding)
      

  return query_embeddings, positive_embeddings

def build_triplets(queries, positives, seed=42):
    rng = np.random.default_rng(seed)
    N = len(queries)
    
    queries = np.asarray(queries)
    positives = np.asarray(positives)
    

    # Shuffle queries and positives together
    indices = rng.permutation(N)
    queries = queries[indices]
    positives = positives[indices]
    
    #create a randomised copy of the negatives
    negatives = positives.copy()
    rng.shuffle(negatives)

    # Fix negative[i] == positive[i] that will degrade the model
    # Use roll and swap if needed
    conflicts = np.where(np.all(negatives == positives, axis=1))[0]
    print(f"{len(conflicts)} found")
    if len(conflicts) > 0:
        negatives = negatives.copy()
        negatives[conflicts] = np.roll(negatives, 1, axis=0)[conflicts]

    return {
        "query": queries,
        "pos": positives,
        "neg": negatives
    }

if __name__ == "__main__":
  hfapi = HfApi(token=os.getenv("HF_TOKEN"))

  print ("loading word2vec model...")
  model = api.load("word2vec-google-news-300")
  print ("word2vec model loaded")

  for filename in ["test-00000-of-00001.parquet", "train-00000-of-00001.parquet","validation-00000-of-00001.parquet"]:
    print(f"computing embeddings for '{filename}'")
    query_embeddings, positive_embeddings = compute_embeddings(filename, model)
    print("computing embeddings complete")

    print("building triplets")
    triplets = build_triplets(query_embeddings, positive_embeddings)
    print("building triplets complete")

    pickle_filename = f"{filename}.triplet.embeddings.pkl" 
    print(f"saving triplets embeddings to '{pickle_filename}'")
    with open(pickle_filename, "wb") as f:
      pickle.dump(triplets, f)
    print("saving pickle complete")

    print(f"uploading '{pickle_filename}' to huggingface")
    hfapi.upload_file(
      path_or_fileobj=pickle_filename,
      path_in_repo=pickle_filename,
      repo_id="danbhf/two-towers",
      repo_type="dataset",  # or "model" depending on your use
    )
    print("upload complete")
    
# Example Load
#with open("vectors.pkl", "rb") as f:
#    triplets = pickle.load(f)