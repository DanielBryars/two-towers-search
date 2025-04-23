import torch
import torch.nn as nn
import os
import numpy as np
import sys
import torch
import os
import pandas as pd
from tqdm import tqdm
import sys
import pickle
import os
import gensim.downloader as api
from gensim.models import KeyedVectors 
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.typing import NDArray
from tqdm import tqdm

EMBEDDING_DIM = 300
OUTPUT_DIM = 128

class DocTower(nn.Module):
    def __init__(self, input_dim=EMBEDDING_DIM, output_dim=OUTPUT_DIM):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.fc(x)
    
class QueryTower(nn.Module):
    def __init__(self, input_dim=EMBEDDING_DIM, output_dim=OUTPUT_DIM):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

def text_to_embedding(text: str, model: KeyedVectors) -> NDArray[np.float32]:
    words = text.lower().split()
    vectors = [model[word] for word in words if word in model]
    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0).astype(np.float32)

def load_checkpoint(checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"loading model checkpoint:'{checkpoint_path}'")
    docModel = DocTower()
    queryModel = QueryTower()

    checkpoint = torch.load(checkpoint_path, map_location='cpu')  # or 'cuda' if using GPU
    queryModel.load_state_dict(checkpoint['queryModel'])
    docModel.load_state_dict(checkpoint['docModel'])
    
    queryModel.eval()
    docModel.eval()
    return queryModel, docModel

if __name__ == '__main__':
    
    query_tower = QueryTower()
    doc_tower = DocTower()

    q = torch.randn(4, EMBEDDING_DIM)
    d = torch.randn(4, EMBEDDING_DIM)
    labels = torch.tensor([1, 1, -1, -1], dtype=torch.float32)  # 1 for match, -1 for non-match

    # Forward
    q_emb = query_tower(q)
    d_emb = doc_tower(d)

    # Loss
    print(q_emb.shape, d_emb.shape)

    print(nn.CosineSimilarity()(q_emb, d_emb).item()) #Should be ?

    loss_fn = nn.CosineEmbeddingLoss()
    loss = loss_fn(q_emb, d_emb, labels)

    print(f"CosineEmbeddingLoss (random input): {loss.item():.4f}")  # Expect ~1.0 with random vectors
    loss.backward()
    print("Backprop OK")



'''
def neural_network(token_embedding, hidden state):
    hidden_state, prediction = some_function(token_embedding, hidden state)
    return hidden_state, prediction

def RNN(token_embeddings, initial_hidden_state):
    hidden_state = initial_hidden_state
    for token_embedding in token_embeddings:
        hidden_state, prediction = neural_network(token_embedding, hidden_state)
    final_hidden_state = hidden_state
    final_prediction = prediction 
    return final_hidden_state, final_prediction
'''
