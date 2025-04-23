import wandb
import torch
import dataset
import datetime
import model
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import wandb
import time
import os
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.nn import TripletMarginLoss

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(queryModel, docModel, dataloader, optimizer, device, epoch, loss_fn, step_offset=0):
    queryModel.train()
    docModel.train()    
    step = step_offset

    loop = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch in loop:
        query_input, pos_doc_input, neg_doc_input = [x.to(device) for x in batch]

        query_emb = F.normalize(queryModel(query_input), dim=-1)
        pos_doc_emb = F.normalize(docModel(pos_doc_input), dim=-1)
        neg_doc_emb = F.normalize(docModel(neg_doc_input), dim=-1)

        loss = loss_fn(query_emb, pos_doc_emb, neg_doc_emb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({'train/loss': loss.item()}, step=step)
        loop.set_postfix(loss=loss.item())
        step += 1

    return step

def evaluate(queryModel, docModel, dataloader, device, loss_fn, epoch=None, step=None):
    queryModel.eval()
    docModel.eval()
    
    total_loss = 0.0
    total_batches = 0

    loop = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", leave=False)
    with torch.no_grad():
        for batch in loop:
            query_input, pos_doc_input, neg_doc_input = [x.to(device) for x in batch]

            query_emb = F.normalize(queryModel(query_input), dim=-1)
            pos_doc_emb = F.normalize(docModel(pos_doc_input), dim=-1)
            neg_doc_emb = F.normalize(docModel(neg_doc_input), dim=-1)

            loss = loss_fn(query_emb, pos_doc_emb, neg_doc_emb)

            total_loss += loss.item()
            total_batches += 1
            loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / total_batches if total_batches > 0 else float('nan')
    if step is not None:
        wandb.log({'val/loss': avg_loss}, step=step)
    return avg_loss

def save_checkpoint(queryModel, docModel, epoch, ts):
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_name = f'{ts}.{epoch + 1}.twotower.pth'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    torch.save({
        'queryModel': queryModel.state_dict(),
        'docModel': docModel.state_dict(),
        'epoch': epoch,
    }, checkpoint_path)

    # Create wandb artifact and log it
    artifact = wandb.Artifact('model-weights', type='model', description='Two-tower model weights')
    artifact.add_file(checkpoint_path)
    wandb.log_artifact(artifact)
    print(f"Checkpoint saved at {checkpoint_path}")

dataset.download_pickles_from_hugginface()
set_seed()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')


hyperparameters = {
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'batch_size': 512,
        'num_epochs': 10,        
        'word2vec_dim': 300,
        'embedding_dim': 128,
        'loss_function': 'contrastive_cosine_loss',#'nn_triplet_margin_loss',
        'margin':0.2,
        'p':2,
        'patience': 3
}

def contrastive_cosine_loss(query, positive, negative, margin=0.2):
    """
    Computes contrastive loss based on cosine similarity.

    Loss = max(0, margin - cos(q, d⁺) + cos(q, d⁻))

    Args:
        query (Tensor): Query embeddings, shape (batch_size, dim)
        positive (Tensor): Positive document embeddings, shape (batch_size, dim)
        negative (Tensor): Negative document embeddings, shape (batch_size, dim)
        margin (float): Margin value

    Returns:
        Tensor: Scalar loss
    """
    sim_pos = F.cosine_similarity(query, positive)
    sim_neg = F.cosine_similarity(query, negative)
    loss = F.relu(margin - sim_pos + sim_neg)
    return loss.mean()

#loss_fn = TripletMarginLoss(margin=hyperparameters['margin'], p=hyperparameters['p'])

loss_fn = contrastive_cosine_loss

wandb.init(
    project='mlx7-week2-two-towers',
    name=f'{ts}',
    config=hyperparameters
)

train_dataset = dataset.TripletDataset("train-00000-of-00001.parquet.triplet.embeddings.pkl")
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=hyperparameters['batch_size'])

val_dataset = dataset.TripletDataset("validation-00000-of-00001.parquet.triplet.embeddings.pkl")
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=hyperparameters['batch_size'])


queryModel = model.QueryTower()
docModel = model.DocTower()

queryModel.to(device)
docModel.to(device)

print('queryModel:params', sum(p.numel() for p in queryModel.parameters()))
print('docModel:params', sum(p.numel() for p in docModel.parameters()))

optimizer = torch.optim.Adam(
    list(queryModel.parameters()) + list(docModel.parameters()), 
    lr=hyperparameters['learning_rate'], 
    weight_decay=hyperparameters['weight_decay']
)

step = 0
best_val_loss = float('inf')
epochs_no_improve = 0

patience= hyperparameters['patience']

for epoch in range(1, hyperparameters['num_epochs'] + 1):
  step = train_one_epoch(queryModel, docModel, train_loader, optimizer, device, epoch, loss_fn, step_offset=step)
  val_loss = evaluate(queryModel, docModel, val_loader, device, loss_fn, epoch=epoch, step=step)

  print(f"Epoch {epoch} complete | Val Loss: {val_loss:.4f}")

  if val_loss < best_val_loss:
    best_val_loss = val_loss
    epochs_no_improve = 0
    save_checkpoint(queryModel, docModel, epoch, ts)
  else:
    epochs_no_improve += 1
    print(f"No improvement. Early stop patience: {epochs_no_improve}/{patience}")

  if epochs_no_improve >= patience:
    print("Early stopping triggered.")
    break

wandb.finish()