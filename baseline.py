from load import *
from model import *
import torch
from tqdm import tqdm
from torch_geometric.data import DataLoader
from torch_geometric.data import Batch
from transformers import AutoModelWithLMHead, AutoTokenizer



if __name__ == "__main__":
    print("loading dataset...")
    lm_qm9 = LM_QM9(root='data/ModifiedQM9')
    print("finish loading")
    num_epochs = 100

    print("loading training dataset...")
    train_lm_qm9 = lm_qm9[:110000].copy()
    val_lm_qm9 = lm_qm9[110000:120000].copy()
    test_lm_qm9 = lm_qm9[120000:].copy()

    pretrain_chemberta = AutoModelWithLMHead.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    property_model = ChemBERTaForPropertyPrediction(pretrain_chemberta)

    
    train_loader = DataLoader(train_lm_qm9, batch_size=32, shuffle=True)
    loss_func = nn.MSELoss()

    print("start traning")
    optimizer = torch.optim.Adam(property_model.parameters(), lr=1e-4)
    for epoch in tqdm(range(num_epochs)):
        for batch in train_loader:
            print(batch.x)
            outputs = property_model(batch.x, batch.attention_mask)
            loss = loss_func(outputs, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
