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

from model import *


from huggingface_hub import HfApi



if __name__ == "__main__":
  if len(sys.argv[1:]) > 0:
     query = sys.argv[1]
  else:
     query = "How do I make an omlete?"

  hfapi = HfApi(token=os.getenv("HF_TOKEN"))

  print ("loading word2vec model...")
  w2v_model = api.load("word2vec-google-news-300")
  print ("word2vec model loaded")

  queryModel, _ = load_checkpoint("checkpoints/2025_04_23__12_47_41.9.twotower.pth")

    

