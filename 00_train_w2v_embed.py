from gensim.models import KeyedVectors
from datasets import load_dataset
import numpy as np
import os
import tqdm
import pickle
import gensim.downloader as api

import passages_parser


model = api.load("word2vec-google-news-300")

def text_to_embedding(text, model):
    words = text.lower().split()
    vectors = [model[word] for word in words if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# Load MS MARCO (train split as example)
dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train")

# Precompute embeddings
embedding_dir = "embeddings"
os.makedirs(embedding_dir, exist_ok=True)

query_vectors = {}
passage_vectors = {}

for sample in tqdm.tqdm(dataset, desc="Computing embeddings"):
    qid = sample['query_id']
    passages_json = sample['passages']
    query = sample['query']

    selected_passages, not_selected_passages = passages_parser.parse__passages_json(passages_json)

    if len(selected_passages) > 0:
        query_vectors[qid] = text_to_embedding(query, model)    
        passage_vectors[qid] = text_to_embedding(selected_passages[0], model)

# Save
with open(os.path.join(embedding_dir, 'query_vectors.pkl'), 'wb') as f:
    pickle.dump(query_vectors, f)

with open(os.path.join(embedding_dir, 'passage_vectors.pkl'), 'wb') as f:
    pickle.dump(passage_vectors, f)
