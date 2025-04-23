import torch
import model

import torch
import os

def load_checkpoint(checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    docModel = model.DocTower()
    queryModel = model.QueryTower()

    checkpoint = torch.load(checkpoint_path, map_location='cpu')  # or 'cuda' if using GPU
    queryModel.load_state_dict(checkpoint['queryModel'])
    docModel.load_state_dict(checkpoint['docModel'])
    
    return queryModel, docModel


queryModel, docModel = load_checkpoint("checkpoint/2025_04_23__12_47_41.9.twotower.pth")

print(queryModel)
print(docModel)

#


#bryars-bryars/mlx7-week2-two-towers/model-weights:v24


