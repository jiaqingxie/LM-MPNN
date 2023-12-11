from load import *
from model import *
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from transformers import AutoModelWithLMHead

def test(loader, model):
    model.eval()
    error = 0

    for data in loader:
        data = data.to("cuda")
        error += (model(data.x, data.attention_mask) * std - data.y[:, target] * std).abs().sum().item()  # MAE
    return error / len(loader.dataset)


if __name__ == "__main__":
    print("loading dataset...")
    name = "ESOL"
    lm_mol = LM_MoleculeNet(root='data/ModifiedMol/{}'.format(name), name=name).shuffle()
    print("finish loading")
    target = 0 
    num_epochs = 100
    std = lm_mol.y[:, 0].std(dim=0, keepdim=True).to("cuda")

    len_dataset = len(lm_mol)

    print("loading training dataset...")
    val_lm_mol = lm_mol[: int(len_dataset * 0.1)].copy()
    test_lm_mol = lm_mol[int(len_dataset * 0.1) :int(len_dataset * 0.2)].copy()
    train_lm_mol = lm_mol[int(len_dataset * 0.2):].copy()

    pretrain_chemberta = AutoModelWithLMHead.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    property_model = ChemBERTaForPropertyPrediction(pretrain_chemberta).to("cuda")

    
    train_loader = DataLoader(train_lm_mol, batch_size=128, shuffle=True)
    valid_loader = DataLoader(val_lm_mol, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_lm_mol, batch_size=128, shuffle=False)

    print("start traning")
    optimizer = torch.optim.Adam(property_model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.7, patience=5,
                                                       min_lr=0.00001)
    for epoch in tqdm(range(num_epochs)):
        lr = scheduler.optimizer.param_groups[0]['lr']
        property_model.train()
        loss_all = 0
        for batch in train_loader:
            batch = batch.to("cuda")
            outputs = property_model(batch.x, batch.attention_mask)
            loss = F.mse_loss(outputs, batch.y[:, 0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_all += loss.item() * batch.num_graphs
        train_loss = loss_all / len(train_loader.dataset)

        valid_error = test(valid_loader, property_model)
        scheduler.step(valid_error)
        test_error = test(test_loader, property_model)

        print(f'Epoch: {epoch:03d}, LR: {lr:7f}, Loss: {loss:.7f}, '
            f'Val MAE: {valid_error:.7f}, Test MAE: {test_error:.7f}')