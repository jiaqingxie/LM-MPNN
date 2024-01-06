# GICL
Geometry Injection of Molecule with Constrastive Learning


## Baseline: Pretrained Mol-Language Model
### Chemberta

#### BBBP on CUDA
```
python baseline_MolNet.py --device cuda --pretrained v1 --dataset BBBP --out_dim 2 --task clf
```

#### ESOL on CUDA
```
python baseline_MolNet.py --device cuda --pretrained v1 --dataset ESOL --out_dim 1 --task reg
```

### Chemberta-v2

#### BBBP on CUDA
```
python baseline_MolNet.py --device cuda --pretrained v2 --dataset BBBP --out_dim 2 --task clf
```

#### ESOL on CUDA
```
python baseline_MolNet.py --device cuda --pretrained v2 --dataset ESOL --out_dim 1 --task reg
```