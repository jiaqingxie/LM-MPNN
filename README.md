# GICL

Geometry Injection of Molecule with Contrastive Learning


### Baseline: Pretrained Mol-Language Model
**HIV**: ```python baseline.py --dataset HIV --out_dim 2 --task clf --device cuda```   
**BACE**: ```python baseline.py --dataset BACE --out_dim 2 --task clf --device cuda```   
**BBBP**: ```python baseline.py --dataset BBBP --out_dim 2 --task clf --device cuda```   
**ESOL**: ```python baseline.py --dataset ESOL --out_dim 1 --task reg --device cuda```   
**QM9**: ```python baseline.py --dataset QM9 --out_dim 1 --task reg --device cuda```   
--pretrained: v1/v2/gpt   

### Baseline: Message Passing Neural Networks (GNN)
**HIV**: ```python gnn.py --dataset HIV --out_dim 2 --task clf --device cuda```   
**BACE**: ```python gnn.py --dataset BACE --out_dim 2 --task clf --device cuda```   
**BBBP**: ```python gnn.py --dataset BBBP --out_dim 2 --task clf --device cuda```   
**ESOL**: ```python gnn.py --dataset ESOL --out_dim 1 --task reg --device cuda```   
**QM9**: ```python gnn.py --dataset QM9 --out_dim 1 --task reg --device cuda```   

### Node-Level Contrastive Learning
**HIV**: ```python node_contrast.py --dataset HIV --out_dim 2 --task clf --device cuda```   
**BACE**: ```python node_contrast.py --dataset BACE --out_dim 2 --task clf --device cuda```   
**BBBP**: ```python node_contrast.py --dataset BBBP --out_dim 2 --task clf --device cuda```   
**ESOL**: ```python node_contrast.py --dataset ESOL --out_dim 1 --task reg --device cuda```   
**QM9**: ```python node_contrast.py --dataset QM9 --out_dim 1 --task reg --device cuda```   
--pretrained: v1/v2/gpt   
--weight_cl: a float number   

### Graph-Level Contrastive Learning
**HIV**: ```python graph_contrast.py --dataset HIV --out_dim 2 --task clf --device cuda```   
**BACE**: ```python graph_contrast.py --dataset BACE --out_dim 2 --task clf --device cuda```   
**BBBP**: ```python graph_contrast.py --dataset BBBP --out_dim 2 --task clf --device cuda```   
**ESOL**: ```python graph_contrast.py --dataset ESOL --out_dim 1 --task reg --device cuda```   
**QM9**: ```python graph_contrast.py --dataset QM9 --out_dim 1 --task reg --device cuda```   
--pretrained: v1/v2/gpt   
--weight_cl: a float number   

### Late Fusion
**HIV**: ```python late_fusion.py --dataset HIV --out_dim 2 --task clf --device cuda```   
**BACE**: ```python late_fusion.py --dataset BACE --out_dim 2 --task clf --device cuda```   
**BBBP**: ```python late_fusion.py --dataset BBBP --out_dim 2 --task clf --device cuda```   
**ESOL**: ```python late_fusion.py --dataset ESOL --out_dim 1 --task reg --device cuda```   
**QM9**: ```python late_fusion.py --dataset QM9 --out_dim 1 --task reg --device cuda```   
--pretrained: v1/v2/gpt   
--aggr: sum/max/concat/gate   
--graph_model: mpnn/gnn   

### Joint Fusion
**HIV**: ```python joint_fusion.py --dataset HIV --out_dim 2 --task clf --device cuda```   
**BACE**: ```python joint_fusion.py --dataset BACE --out_dim 2 --task clf --device cuda```   
**BBBP**: ```python joint_fusion.py --dataset BBBP --out_dim 2 --task clf --device cuda```   
**ESOL**: ```python joint_fusion.py --dataset ESOL --out_dim 1 --task reg --device cuda```   
**QM9**: ```python joint_fusion.py --dataset QM9 --out_dim 1 --task reg --device cuda```   
--pretrained: v1/v2/gpt   
--joint: gnn2lm/lm2gnn   
--aggr: sum/max/concat   
--aggr: mpnn/gnn   
