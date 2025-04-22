import torch
import torch.nn.functional as F
import torch.nn as nn

# Example vectors
x1 = torch.tensor([[1.0, 0.0],     # perfectly aligned with x2
                   [1.0, 0.0]])    # same as above

x2 = torch.tensor([[1.0, 0.0],     # same as x1 (cos = 1)
                   [-1.0, 0.0]])   # opposite of x1 (cos = -1)

labels = torch.tensor([1.0, -1.0])  # 1 = match, -1 = no match

loss_fn = nn.CosineEmbeddingLoss(margin=0.0)
loss = loss_fn(x1, x2, labels)

print("Loss:", loss.item())
