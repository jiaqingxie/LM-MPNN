import torch
from tqdm import tqdm
from torch_geometric.datasets import QM9
from torch_geometric.data import InMemoryDataset, Data
from transformers import AutoTokenizer

### QM9 encoded with tokens
class LM_QM9(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pretrain_model = "ChemBERTa-zinc-base-v1", padding = True):
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
        smiles_list = [data.smiles for data in qm9]
        tokenized_features = self.pretrain(smiles_list, self.pretrain_model)

        data_list = []
        for idx, data in enumerate(qm9):
            data.x = tokenized_features[idx]
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def pretrain(self, smiles_list, pretrain_model):
        tokenizer = AutoTokenizer.from_pretrained("seyonec/{}".format(pretrain_model))
        tokenized_features = tokenizer(smiles_list, padding=self.padding, truncation=True, return_tensors='pt')
        return tokenized_features['input_ids']

# Usage
if __name__ == "__main__":
    modified_dataset = LM_QM9(root='data/ModifiedQM9')
    print(modified_dataset[2].x)