from load import *
from model import *
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.data import Batch
from transformers import AutoModelWithLMHead, AutoTokenizer


def test(loader, model):
    model.eval()
    error = 0

    for data in loader:
        data = data.to("cuda")
        error += (model(data) * std - data.y * std).abs().sum().item()  # MAE
    return error / len(loader)


if __name__ == "__main__":
    print("loading dataset...")
    lm_qm9 = LM_QM9(root='data/ModifiedQM9').shuffle()
    print("finish loading")
    target = 0 
    num_epochs = 100
    std = lm_qm9.y.std(dim=0, keepdim=True)
  

    print("loading training dataset...")
    val_lm_qm9 = lm_qm9[:10000].copy()
    test_lm_qm9 = lm_qm9[10000:20000].copy()
    train_lm_qm9 = lm_qm9[20000:].copy()

    pretrain_chemberta = AutoModelWithLMHead.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    property_model = ChemBERTaForPropertyPrediction(pretrain_chemberta).to("cuda")

    
    train_loader = DataLoader(train_lm_qm9, batch_size=128, shuffle=True)
    valid_loader = DataLoader(val_lm_qm9, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_lm_qm9, batch_size=128, shuffle=False)

    print("start traning")
    optimizer = torch.optim.Adam(property_model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.7, patience=5,
                                                       min_lr=0.00001)
    for epoch in range(num_epochs):
        for batch in tqdm(train_loader):
            property_model.train()
            batch = batch.to("cuda")
            outputs = property_model(batch.x, batch.attention_mask)
            loss = F.mse_loss(outputs, batch.y[:, target])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(loss)
        valid_error =  test(valid_loader, property_model)
        print(valid_error)
        scheduler.step(valid_error)
