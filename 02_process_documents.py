import torch
import model
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

from model import DocTower, QueryTower


from huggingface_hub import HfApi

#curl -L -O https://huggingface.co/datasets/microsoft/ms_marco/resolve/main/v1.1/test-00000-of-00001.parquet
#curl -L -O https://huggingface.co/datasets/microsoft/ms_marco/resolve/main/v1.1/train-00000-of-00001.parquet
#curl -L -O https://huggingface.co/datasets/microsoft/ms_marco/resolve/main/v1.1/validation-00000-of-00001.parquet


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


def text_to_embedding(text: str, model: KeyedVectors) -> NDArray[np.float32]:
    words = text.lower().split()
    vectors = [model[word] for word in words if word in model]
    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0).astype(np.float32)

if __name__ == "__main__":
  if len(sys.argv[1:]) > 0:
     filepath = sys.argv[1]
  else:
     filepath = "checkpoints/2025_04_23__12_47_41.9.twotower.pth"

  hfapi = HfApi(token=os.getenv("HF_TOKEN"))

  print ("loading word2vec model...")
  w2v_model = api.load("word2vec-google-news-300")
  print ("word2vec model loaded")

  queryModel, docModel = load_checkpoint(filepath)

  #Get all the documents and compute the 
  #parquet_filenames = ["test-00000-of-00001.parquet", "train-00000-of-00001.parquet","validation-00000-of-00001.parquet"]

  parquet_filenames = ["test-00000-of-00001.parquet"]

  #put all these files into memory 
  total_document_count = 0
  print("Counting documents to preallocate memory")
  for parquet_filename in tqdm(parquet_filenames, desc="Counting files"):
    df = pd.read_parquet(parquet_filename)
    count = sum(len(row.passages['passage_text']) for row in df.itertuples(index=False)) # type: ignore
    print(f"{count} documents in {parquet_filename}")
    total_document_count += count

  print(f"total count {total_document_count}")

  embeddings = np.empty((total_document_count, 128), dtype=np.float32)
  documents = []
  
  i = 0

  for parquet_filename in tqdm(parquet_filenames, desc="Embedding files"):
    df = pd.read_parquet(parquet_filename)  
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"Processing {parquet_filename}", leave=False):
      for passage in row.passages['passage_text']: # type: ignore
        passage_embedding = text_to_embedding(passage, w2v_model) # type: ignore
        with torch.no_grad():
          passage_tensor = torch.from_numpy(passage_embedding).unsqueeze(0)  # shape (1, 300)
          docModel_embedding = docModel(passage_tensor).squeeze(0).numpy()
          embeddings[i] = docModel_embedding
        documents.append(passage)
        i+=1

  pickle_filename = "documentEmbeddings.testonly.pkl"
  with open(pickle_filename, "wb") as f:
    pickle.dump((embeddings, documents), f)

    print(f"uploading '{pickle_filename}' to huggingface")
    hfapi.upload_file(
      path_or_fileobj=pickle_filename,
      path_in_repo=pickle_filename,
      repo_id="danbhf/two-towers",
      repo_type="dataset",  # or "model" depending on your use
    )
    print("upload complete")

