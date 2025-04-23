import torch
import pickle


#https://huggingface.co/datasets/microsoft/ms_marco



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


class Wiki(torch.utils.data.Dataset):
  def __init__(self):
    self.vocab_to_int = pickle.load(open('./tkn_words_to_ids.pkl', 'rb'))
    self.int_to_vocab = pickle.load(open('./tkn_ids_to_words.pkl', 'rb'))
    self.corpus = pickle.load(open('./corpus.pkl', 'rb'))
    self.tokens = [self.vocab_to_int[word] for word in self.corpus]

  def __len__(self):
    return len(self.tokens)

  def __getitem__(self, idx: int):
    ipt = self.tokens[idx]
    prv = self.tokens[idx-2:idx]
    nex = self.tokens[idx+1:idx+3]
    if len(prv) < 2: prv = [0] * (2 - len(prv)) + prv
    if len(nex) < 2: nex = nex + [0] * (2 - len(nex))
    return torch.tensor(prv + nex), torch.tensor([ipt])


#
#
#
if __name__ == '__main__':
  ds = Wiki()
  print(ds.tokens[:15])
  # print(ds[0])
  print(ds[5])
  dl = torch.utils.data.DataLoader(dataset=ds, batch_size=3)
  ex = next(iter(dl))
  print(ex)