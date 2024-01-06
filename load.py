import torch
from tqdm import tqdm
from torch_geometric.datasets import QM9, MoleculeNet
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils.smiles import from_smiles, to_smiles
from transformers import AutoTokenizer

### QM9 encoded with tokens
class LM_QM9(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pretrain_model="DeepChem/ChemBERTa-10M-MTR", padding=True):
        self.pretrain_model = pretrain_model
        self.padding = padding
        super(LM_QM9, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return ['some_file_name']

    @property
    def processed_file_names(self):
        return ['lm_qm9.pt']

    def download(self):
        pass

    def process(self):
        qm9 = QM9(root=self.root)

        temp_list, smiles_list = [], []
        for data in qm9:
            temp_list.append(from_smiles(data.smiles))
            smiles_list.append(data.smiles)
        input_ids = self.pretrain(smiles_list, self.pretrain_model)

        data_list = []
        for idx, data in enumerate(temp_list):
            data.x = data.x.to(dtype=torch.float32)
            data.edge_attr = data.edge_attr.to(dtype=torch.float32)
            data.y = qm9[idx].y
            data.input_ids = input_ids[idx].unsqueeze(0)
            data.attention_mask = ~(data.input_ids == 0)
            data.mol_mask = torch.isin(data.input_ids, torch.tensor([16, 15, 23, 25, 19, 44, 27]))  # C c N n O o F
            if (data.x.shape[0] == 0) or (data.x.shape[0] != data.mol_mask.sum()):
                continue
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def pretrain(self, smiles_list, pretrain_model):
        tokenizer = AutoTokenizer.from_pretrained(pretrain_model)
        tokenized_features = tokenizer.batch_encode_plus(smiles_list, padding=self.padding, truncation=True, return_tensors='pt')
        return tokenized_features['input_ids']


### MoleculeNet encoded with tokens
class LM_MoleculeNet(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pretrain_model="DeepChem/ChemBERTa-10M-MTR", padding=True, name="ESOL"):
        self.pretrain_model = pretrain_model
        self.padding = padding
        self.name = name
        super(LM_MoleculeNet, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return ['some_file_name']

    @property
    def processed_file_names(self):
        return ['lm_molnet.pt']

    def download(self):
        pass

    def process(self):
        mol = MoleculeNet("data/{}".format(self.name), self.name)

        temp_list, smiles_list = [], []
        for data in mol:
            temp_list.append(from_smiles(data.smiles))
            smiles_list.append(data.smiles)
        input_ids = self.pretrain(smiles_list, self.pretrain_model)

        data_list = []
        for idx, data in enumerate(temp_list):
            data.x = data.x.to(dtype=torch.float32)
            data.edge_attr = data.edge_attr.to(dtype=torch.float32)
            data.y = mol[idx].y
            data.input_ids = input_ids[idx].unsqueeze(0)
            data.attention_mask = ~(data.input_ids == 0)
            data.mol_mask = torch.isin(data.input_ids, torch.tensor([16, 15, 23, 25, 19, 44, 27]))  # C c N n O o F
            if (data.x.shape[0] == 0) or (data.x.shape[0] != data.mol_mask.sum()):
                continue
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def pretrain(self, smiles_list, pretrain_model):
        tokenizer = AutoTokenizer.from_pretrained(pretrain_model)
        tokenized_features = tokenizer.batch_encode_plus(smiles_list, padding=self.padding, truncation=True, return_tensors='pt')
        return tokenized_features['input_ids']