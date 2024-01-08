# GICL

Geometry Injection of Molecule with Contrastive Learning


### Baseline: Pretrained Mol-Language Model
**HIV**: ```python baseline.py --dataset HIV --out_dim 2 --task clf --device cuda --pretrained v1```   
**BACE**: ```python baseline.py --dataset BACE --out_dim 2 --task clf --device cuda --pretrained v1```   
**BBBP**: ```python baseline.py --dataset BBBP --out_dim 2 --task clf --device cuda --pretrained v1```   
**ESOL**: ```python baseline.py --dataset ESOL --out_dim 1 --task reg --device cuda --pretrained v1```   
**QM9**: ```python baseline.py --dataset QM9 --out_dim 1 --task reg --device cuda --pretrained v1```   
--device: cpu / cuda   
--pretrained: v1 / v2 / gpt

### Baseline: Message Passing Neural Networks (GNN)
**HIV**: ```python gnn.py --dataset HIV --out_dim 2 --task clf --device cuda```   
**BACE**: ```python gnn.py --dataset BACE --out_dim 2 --task clf --device cuda```   
**BBBP**: ```python gnn.py --dataset BBBP --out_dim 2 --task clf --device cuda```   
**ESOL**: ```python gnn.py --dataset ESOL --out_dim 1 --task reg --device cuda```   
**QM9**: ```python gnn.py --dataset QM9 --out_dim 1 --task reg --device cuda```   
--device: cpu / cuda

### Method XXX


### Method XXX


### Method XXX


### Method XXX