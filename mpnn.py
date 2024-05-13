from load import *
from model import *
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
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
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--dataset', type=str, default='ESOL', help='Dataset name')
    parser.add_argument('--test_size', type=float, default=0.1, help='test ratio')
    parser.add_argument('--valid_size', type=float, default=0.1, help='valid ratio')
    parser.add_argument('--task', type=str, default='reg', help='reg or clf')
    parser.add_argument('--target', type=int, default=0, help='target index of y')
    parser.add_argument('--mpnn_hidden_dim', type=int, default=64, help='mpnn hidden dimension')
    parser.add_argument('--embed_dim', type=int, default=384, help='embedding dimension')
    parser.add_argument('--out_dim', type=int, default=1, help='number of classes')

    parser.add_argument('--seed', type=int, default=7, help='seed')

    parser.add_argument('--graph_model', type=str, default='mpnn', help='mpnn / gnn')
    args = parser.parse_args()


    args = parser.parse_args()
    torch.manual_seed(args.seed)
    # dataset
    print("Loading and preprocessing dataset ...")
    dataset = None
    if args.dataset == 'QM9':
        dataset = LM_QM9(root='data/ModifiedQM9').shuffle()
    elif args.dataset in ["BACE", "BBBP", "HIV", "ESOL"]:
        dataset = LM_MoleculeNet(root='data/ModifiedMol/{}'.format(args.dataset), name=args.dataset).shuffle()
        print(dataset[0])
    target_std = None
    if args.task == "reg":
        mean, std = dataset.data.y.mean(dim=0, keepdim=True), dataset.data.y.std(dim=0, keepdim=True)
        target_mean, target_std = mean[:, args.target].to(args.device), std[:, args.target].to(args.device)
        dataset.data.y = (dataset.data.y - mean) / std

    if args.dataset == 'QM9':
        test_dataset = dataset[:10000]
        valid_dataset = dataset[10000:20000]
        train_dataset = dataset[20000:]
    elif args.dataset in ["BACE", "BBBP", "HIV", "ESOL", ""]:
        valid_dataset = dataset[:int(len(dataset) * args.valid_size)]
        test_dataset = dataset[int(len(dataset) * args.valid_size):int(len(dataset) * (args.test_size + args.valid_size))]
        train_dataset = dataset[int(len(dataset) * (args.test_size + args.valid_size)):]
    print("Finished loading. ")

    # dataloader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # model
    if args.graph_model == 'mpnn':
        model = NNConvModel(num_features=dataset.num_features,
                            hidden_dim=args.mpnn_hidden_dim,
                            embed_dim=args.embed_dim,
                            out_dim=args.out_dim,
                            task=args.task).to(args.device)
    elif args.graph_model == 'gnn':
        model = GCNModel(num_features=dataset.num_features,
                         hidden_dim=args.mpnn_hidden_dim,
                         embed_dim=args.embed_dim,
                         out_dim=args.out_dim,
                         task=args.task).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.7, patience=5,
                                                           min_lr=0.00001)

    # training loop
    print("Started training ...")
    if args.task == "reg":
        best_valid_metric, best_test_metric = 1e4, 1e4
    elif args.task == "clf":
        best_valid_metric, best_test_metric = 0, 0

    for epoch in range(args.epochs):
        lr = scheduler.optimizer.param_groups[0]['lr']
        model.train()
        loss_all = 0

        for batch in train_loader:
            batch = batch.to(args.device)
            outputs, _, _ = model(batch)
            if args.task == "reg":
                loss = F.mse_loss(outputs, batch.y[:, args.target])
            elif args.task == "clf":
                loss = F.nll_loss(outputs, batch.y[:, args.target].type(torch.LongTensor).to(args.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_all += loss.item() * batch.num_graphs

        train_loss = loss_all / len(train_loader.dataset)
        valid_metric = test(valid_loader, model, target_std, args)
        scheduler.step(valid_metric)
        test_metric = test(test_loader, model, target_std, args)

        if args.task == "reg":
            if valid_metric < best_valid_metric:
                best_valid_metric = valid_metric
            if test_metric < best_test_metric:
                best_test_metric = test_metric
            print(f'Epoch: {epoch:03d}, LR: {lr:4f}, Loss: {train_loss:.4f}, '
                  f'Val MAE: {valid_metric:.4f}, Best Val MAE: {best_valid_metric:.4f}, '
                  f'Test MAE: {test_metric:.4f}, Best Test MAE: {best_test_metric:.4f}')
        elif args.task == "clf":
            if valid_metric > best_valid_metric:
                best_valid_metric = valid_metric
            if test_metric > best_test_metric:
                best_test_metric = test_metric
            print(f'Epoch: {epoch:03d}, LR: {lr:4f}, Loss: {train_loss:.4f}, '
                  f'Val ACC: {valid_metric:.4f}, Best Val ACC: {best_valid_metric:.4f}, '
                  f'Test ACC: {test_metric:.4f}, Best Test ACC: {best_test_metric:.4f}')

    print("########################################")
    print("Method: MPNN Baseline")
    print(f"Dataset: {args.dataset}")
    print(f"Task: {args.task}")
    print(f"Number of epochs: {args.epochs}")
    if args.task == "reg":
        print(f"Best Test MAE: {best_test_metric:.4f}")
    elif args.task == "clf":
        print(f"Best Test ACC: {best_test_metric:.4f}")
    print("########################################")
