from load import *
from model import *
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from transformers import AutoModelWithLMHead


def test(loader, model, target, std):
    model.eval()
    error = 0

    for data in loader:
        data = data.to(device)
        error += (model(data.input_ids, data.attention_mask)[0] * std - data.y[:, target] * std).abs().sum().item()  # MAE
    return error / len(loader.dataset)


if __name__ == "__main__":
    # settings
    target = 0
    num_epochs = 100
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

    # model
    pretrain_chemberta = AutoModelWithLMHead.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    property_model = ChemBERTaForPropertyPrediction(pretrain_chemberta).to(device)
    optimizer = torch.optim.Adam(property_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.7, patience=5,
                                                       min_lr=0.00001)

    for epoch in range(num_epochs):
        lr = scheduler.optimizer.param_groups[0]['lr']
        property_model.train()
        loss_all = 0
        for batch in tqdm(train_loader):
            batch = batch.to(device)
            outputs, _ = property_model(batch.input_ids, batch.attention_mask)
            loss = F.mse_loss(outputs, batch.y[:, target])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_all += loss.item() * batch.num_graphs

        train_loss = loss_all / len(train_loader.dataset)
        valid_error = test(valid_loader, property_model, target, target_std)
        scheduler.step(valid_error)
        test_error = test(test_loader, property_model, target, target_std)

        print(f'Epoch: {epoch:03d}, LR: {lr:7f}, Loss: {train_loss:.7f}, '
            f'Val MAE: {valid_error:.7f}, Test MAE: {test_error:.7f}')