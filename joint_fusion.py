import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from transformers import AutoModelWithLMHead
from load import *
from model import *
from tqdm import tqdm

def test(loader, model, target, std):
    model.eval()
    error = 0

    for batch in loader:
        batch = batch.to(device)
        error += (model(batch) * std - batch.y[:, target] * std).abs().sum().item()
    return error / len(loader.dataset)


if __name__ == "__main__":
    # settings
    target = 0
    num_epochs = 100
    pretrain_chemberta = AutoModelWithLMHead.from_pretrained("DeepChem/ChemBERTa-10M-MTR")
    # pretrain_chemberta = AutoModelWithLMHead.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    gnn_hidden_dim, graph_embedding_dim = pretrain_chemberta.config.hidden_size, pretrain_chemberta.config.hidden_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    print("Loading and preprocessing dataset...")
    dataset = LM_QM9(root='data/ModifiedQM9').shuffle()
    mean, std = dataset.data.y.mean(dim=0, keepdim=True), dataset.data.y.std(dim=0, keepdim=True)
    target_mean, target_std = mean[:, target], std[:, target]
    dataset.data.y = (dataset.data.y - mean) / std
    test_dataset = dataset[:10000]
    valid_dataset = dataset[10000:20000]
    train_dataset = dataset[20000:]
    print("Finished loading. ")

    # dataloader
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # joint fusion model
    joint_model = JointFusionModel(pretrain_chemberta,
                                   dataset.num_features,
                                   gnn_hidden_dim,
                                   graph_embedding_dim).to(device)
    joint_optimizer = torch.optim.Adam(joint_model.parameters(), lr=1e-3)
    joint_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(joint_optimizer, mode='min',
                                                                  factor=0.7, patience=5,
                                                                  min_lr=0.00001)

    for epoch in range(num_epochs):
        joint_model.train()
        loss_all = 0
        for batch in tqdm(train_loader):
            batch = batch.to(device)
            joint_pred = joint_model(batch)
            loss = F.mse_loss(joint_pred, batch.y[:, target])
            joint_optimizer.zero_grad()
            loss.backward()
            joint_optimizer.step()
            loss_all += loss.item() * batch.num_graphs

        train_loss = loss_all / len(train_loader.dataset)
        valid_error = test(valid_loader, joint_model, target, target_std)
        joint_scheduler.step(valid_error)
        test_error = test(test_loader, joint_model, target, target_std)

        print(f'Epoch: {epoch:03d}, Loss: {train_loss:.7f}, '
              f'Val MAE: {valid_error:.7f}, Test MAE: {test_error:.7f}')


