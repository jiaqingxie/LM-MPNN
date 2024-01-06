# GICL
Geometry Injection of Molecule with Constrastive Learning


## Baseline: Pretrained Mol-Language Model
### Chemberta

#### HIV on CUDA
```
python baseline.py --device cuda --pretrained v1 --dataset HIV --out_dim 2 --task clf
```

#### BACE on CUDA
```
python baseline.py --device cuda --pretrained v1 --dataset BACE --out_dim 2 --task clf
```

#### BBBP on CUDA
```
python baseline.py --device cuda --pretrained v1 --dataset BBBP --out_dim 2 --task clf
```

#### ESOL on CUDA
```
python baseline.py --device cuda --pretrained v1 --dataset ESOL --out_dim 1 --task reg
```

#### QM9 on CUDA
```
python baseline.py --device cuda --pretrained v1 --dataset QM9 --out_dim 1 --task reg
```


### Chemberta-v2

#### HIV on CUDA
```
python baseline.py --device cuda --pretrained v2 --dataset HIV --out_dim 2 --task clf
```

#### BACE on CUDA
```
python baseline.py --device cuda --pretrained v2 --dataset BACE --out_dim 2 --task clf
```

#### BBBP on CUDA
```
python baseline.py --device cuda --pretrained v2 --dataset BBBP --out_dim 2 --task clf
```

#### ESOL on CUDA
```
python baseline.py --device cuda --pretrained v2 --dataset ESOL --out_dim 1 --task reg
```

#### QM9 on CUDA
```
python baseline.py --device cuda --pretrained v2 --dataset QM9 --out_dim 1 --task reg
```



## Baseline: Message Passing Neural Networks (GNN)
