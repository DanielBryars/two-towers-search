import torch
import pickle
import tempfile
import os
import torch
import pickle
from huggingface_hub import hf_hub_download
from pathlib import Path

#https://huggingface.co/datasets/microsoft/ms_marco

def download_pickles_from_hugginface(filenames = [
   "test-00000-of-00001.parquet.triplet.embeddings.pkl", 
   "train-00000-of-00001.parquet.triplet.embeddings.pkl",
   "validation-00000-of-00001.parquet.triplet.embeddings.pkl"]):
    for filename in filenames:
      if not Path("myfile.pkl").exists():
        hf_hub_download(
          repo_id="danbhf/two-towers",
          repo_type="dataset",  # must match what you used during upload
          filename=filename)
        
class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_pickle):
        
        data = pickle.load(open(path_to_pickle, 'rb'))
        self.q = data["query"]
        self.p = data["pos"]
        self.n = data["neg"]

    def __len__(self):
        return len(self.q)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.q[idx], dtype=torch.float32),
            torch.tensor(self.p[idx], dtype=torch.float32),
            torch.tensor(self.n[idx], dtype=torch.float32)
        )

if __name__ == '__main__':
  def create_dummy_pickle():
    dummy_data = {
        "query": [[1.0, 2.0], [3.0, 4.0]],
        "pos": [[1.1, 2.1], [3.1, 4.1]],
        "neg": [[0.9, 1.9], [2.9, 3.9]],
    }
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
    with open(temp.name, 'wb') as f:
        pickle.dump(dummy_data, f)
    return temp.name

  def test_triplet_dataset_length():
    path = create_dummy_pickle()
    try:
        dataset = TripletDataset(path)
        assert len(dataset) == 2
    finally:
        os.remove(path)

  path = create_dummy_pickle()
  try:
    ds = TripletDataset(path)
    q, p, n = ds[0]

    assert isinstance(q, torch.Tensor)
    assert isinstance(p, torch.Tensor)
    assert isinstance(n, torch.Tensor)
    assert q.shape == p.shape == n.shape == torch.Size([2])
    assert q.dtype == p.dtype == n.dtype == torch.float32

    dl = torch.utils.data.DataLoader(dataset=ds, batch_size=3)
    ex = next(iter(dl))
    print(ex)

  finally:
    os.remove(path)
  
  