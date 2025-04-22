import gensim.downloader as api
import numpy as np
import pandas as pd
import pickle

model = api.load("word2vec-google-news-300")

def text_to_embedding(text, model):
    words = text.lower().split()
    vectors = [model[word] for word in words if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

#test-00000-of-00001.parquet
#train-00000-of-00001.parquet
#validation-00000-of-00001.parquet
def compute_embeddings(filename):
  df = pd.read_parquet(filename)
  query_vectors = {}
  passage_vectors = {}

  for _, row in df.iterrows():
      qid = row['query_id']
      query = row['query']
      positive_matches = row['passages']['passage_text']
      query_vectors[qid] = text_to_embedding(query, model)    
      passage_vectors[qid] = np.array([text_to_embedding(passage, model) for passage in positive_matches])

  return query_vectors, passage_vectors

filename = "test-00000-of-00001.parquet"
query_vectors, passage_vectors = compute_embeddings(filename)

with open(f"{filename}.embeddings.pkl", "wb") as f:
  pickle.dump((query_vectors, passage_vectors), f)

# Load
#with open("vectors.pkl", "rb") as f:
#    query_vectors, passage_vectors = pickle.load(f)