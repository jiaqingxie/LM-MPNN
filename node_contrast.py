import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from transformers import AutoModelWithLMHead
from load import *
from model import *
from tqdm import tqdm
import argparse


def test(loader, model, std, args):
    model.eval()

    if args.task == "reg":
        error = 0
        for data in loader:
            data = data.to(args.device)
            error += (model(data)[0] * std - data.y[:, args.target] * std).abs().sum().item()  # MAE for regression
        return error / len(loader.dataset)
    else:
        acc = 0
        for data in loader:
            data = data.to(args.device)
            acc += (model(data)[0].argmax(dim=1) == data.y[:, args.target]).sum()  # Acc for classification
        return acc / len(loader.dataset)


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', help='cpu / cuda')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs on training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--pretrained', type=str, default='v2', help='Model choice: v1 / v2 / gpt')
    parser.add_argument('--dataset', type=str, default='ESOL', help='Dataset name')
    parser.add_argument('--test_size', type=float, default=0.1, help='test ratio')
    parser.add_argument('--valid_size', type=float, default=0.1, help='valid ratio')
    parser.add_argument('--task', type=str, default='reg', help='reg or clf')
    parser.add_argument('--target', type=int, default=0, help='target index of y')
    parser.add_argument('--out_dim', type=int, default=1, help='number of classes')
    parser.add_argument('--mpnn_hidden_dim', type=int, default=64, help='mpnn hidden dimension')
    parser.add_argument('--weight_cl', type=float, default=1.0, help='weight of contrastive loss')
    parser.add_argument('--seed', type=int, default=7, help='seed')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.pretrained == "v2":
        pretrain_chemberta = AutoModelWithLMHead.from_pretrained("DeepChem/ChemBERTa-10M-MTR")
    elif args.pretrained == "v1":
        pretrain_chemberta = AutoModelWithLMHead.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    elif args.pretrained == "gpt":
        pass
    
    # dataset
    print("Loading and preprocessing dataset ...")
    dataset = None
    if args.dataset == 'QM9':
        dataset = LM_QM9(root='data/ModifiedQM9').shuffle()
    elif args.dataset in ["BACE", "BBBP", "HIV", "ESOL", "FreeSolv"]:
        dataset = LM_MoleculeNet(root='data/ModifiedMol/{}'.format(args.dataset), name=args.dataset).shuffle()
    target_std = None
    if args.task == "reg":
        mean, std = dataset.data.y.mean(dim=0, keepdim=True), dataset.data.y.std(dim=0, keepdim=True)
        target_mean, target_std = mean[:, args.target].to(args.device), std[:, args.target].to(args.device)
        dataset.data.y = (dataset.data.y - mean) / std

    if args.dataset == 'QM9':
        test_dataset = dataset[:10000]
        valid_dataset = dataset[10000:20000]
        train_dataset = dataset[20000:]
    elif args.dataset in ["BACE", "BBBP", "HIV", "ESOL", "FreeSolv"]:
        valid_dataset = dataset[:int(len(dataset) * args.valid_size)]
        test_dataset = dataset[int(len(dataset) * args.valid_size):int(len(dataset) * (args.test_size + args.valid_size))]
        train_dataset = dataset[int(len(dataset) * (args.test_size + args.valid_size)):]
    print("Finished loading. ")

    # dataloader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # BERT model
    bert_model = ChemBERTaForPropertyPrediction(chemberta_model=pretrain_chemberta,
                                                out_dim=args.out_dim,
                                                task=args.task).to(args.device)
    bert_optimizer = torch.optim.Adam(bert_model.parameters(), lr=args.lr)
    bert_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(bert_optimizer, mode='min',
                                                                factor=0.7, patience=5,
                                                                min_lr=0.00001)

    # MPNN model
    mpnn_model = NNConvModel(num_features=dataset.num_features,
                             hidden_dim=args.mpnn_hidden_dim,
                             embed_dim=pretrain_chemberta.config.hidden_size,
                             out_dim=args.out_dim,
                             task=args.task).to(args.device)
    mpnn_optimizer = torch.optim.Adam(mpnn_model.parameters(), lr=args.lr)
    mpnn_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(mpnn_optimizer, mode='min',
                                                                factor=0.7, patience=5,
                                                                min_lr=0.00001)

    # training loop
    print("Started training ...")
    if args.task == "reg":
        best_valid_metric, best_test_metric = 1e4, 1e4
    elif args.task == "clf":
        best_valid_metric, best_test_metric = 0, 0

    for epoch in range(args.epochs):
        bert_model.train()
        mpnn_model.train()
        loss_all = 0
        for batch in train_loader:
            """
            Contrastive idea: Within a batch,
                              bert_node_embed (num_nodes, embed_dim) as anchor,
                              mpnn_node_embed (num_nodes, embed_dim) as positive,
                              neg_mpnn_node_embed (num_nodes, embed_dim) as negative.
            """
            batch = batch.to(args.device)
            bert_pred, _, bert_node_embed = bert_model(batch)
            _, _, mpnn_node_embed = mpnn_model(batch)
            bert_node_embed = bert_node_embed[batch.mol_mask]
            neg_mpnn_node_embed = mpnn_node_embed[torch.randperm(mpnn_node_embed.shape[0])]

            if args.task == "reg":
                target_loss = F.mse_loss(bert_pred, batch.y[:, args.target])
            elif args.task == "clf":
                target_loss = F.nll_loss(bert_pred, batch.y[:, args.target].type(torch.LongTensor).to(args.device))
            contrastive_loss = F.triplet_margin_loss(bert_node_embed, mpnn_node_embed, neg_mpnn_node_embed)
            loss = target_loss + args.weight_cl * contrastive_loss

            bert_optimizer.zero_grad()
            mpnn_optimizer.zero_grad()
            loss.backward()
            bert_optimizer.step()
            mpnn_optimizer.step()
            loss_all += loss.item() * batch.num_graphs
            target_loss_all = target_loss.item() * batch.num_graphs

        train_loss = loss_all / len(train_loader.dataset)
        target_loss = target_loss_all / len(train_loader.dataset)
        valid_metric = test(valid_loader, bert_model, target_std, args)
        bert_scheduler.step(valid_metric)
        mpnn_scheduler.step(valid_metric)
        test_metric = test(test_loader, bert_model, target_std, args)

        if args.task == "reg":
            if valid_metric < best_valid_metric:
                best_valid_metric = valid_metric
            if test_metric < best_test_metric:
                best_test_metric = test_metric
            print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Target Loss: {target_loss:.4f}, '
                  f'Val MAE: {valid_metric:.4f}, Best Val MAE: {best_valid_metric:.4f}, '
                  f'Test MAE: {test_metric:.4f}, Best Test MAE: {best_test_metric:.4f}')
        elif args.task == "clf":
            if valid_metric > best_valid_metric:
                best_valid_metric = valid_metric
            if test_metric > best_test_metric:
                best_test_metric = test_metric
            print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Target Loss: {target_loss:.4f}, '
                  f'Val ACC: {valid_metric:.4f}, Best Val ACC: {best_valid_metric:.4f}, '
                  f'Test ACC: {test_metric:.4f}, Best Test ACC: {best_test_metric:.4f}')

    print("########################################")
    print("Method: Node-Level Contrastive Learning")
    print(f"Dataset: {args.dataset}")
    print(f"Task: {args.task}")
    print(f"Pretrained model: {args.pretrained}")
    print(f"Number of epochs: {args.epochs}")
    print(f"Contrastive loss weight: {args.weight_cl}")
    if args.task == "reg":
        print(f"Best Test MAE: {best_test_metric:.4f}")
    elif args.task == "clf":
        print(f"Best Test ACC: {best_test_metric:.4f}")
    print("########################################")
