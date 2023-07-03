# Seq-HGNN: Learning Sequential Node Representation on Heterogeneous Graph





## Requirements

We recommend employing Conda to install the runtime environment.

- cuda==11.* (In my case, cuda==11.7)

- python==3.10

  ```bash
  conda create -n seq_hgnn python=3.10
  conda activate seq_hgnn
  ```

- pytorch==2.0.1

  ```bash
  conda install pytorch=2.0.1 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
  ```

- pyg==2.3.0

  ```bash
  conda install pyg=2.3.0 -c pyg
  conda install pytorch-sparse -c pyg
  ```

- dgl==0.9.1

  ```bash
  conda install -c dglteam dgl-cuda11.7=0.9.1post1
  ```

- dglke==0.1.0 (install from source)

  ```bash
  pip install git+https://github.com/awslabs/dgl-ke.git#subdirectory=python
  ```

- ogb

  ```bash
  conda install -c conda-forge ogb
  ```

- einops

  ```bash
  conda install -c conda-forge einops
  ```

- torchmetrics==0.10

  ```bash
  conda install -c conda-forge torchmetrics=0.10
  ```

- wandb

  ```bash
  conda install -c conda-forge wandb
  ```

  

## Data preparation

### ogbn-mag

In this part, we refer to [NARS](https://github.com/facebookresearch/NARS) and [SeHGNN](https://github.com/ICT-GIMLab/SeHGNN/tree/master/ogbn).

First, we generate node embeddings using [dgl-ke](https://github.com/awslabs/dgl-ke).

```
cd data/complex_nars
python convert_to_triplets.py --dataset mag
bash train_graph_emb.sh mag
```

Check the running log to find where the generated ComplEx features are saved. For example, if the save folder is `ckpts/ComplEx_mag_0`, run

```bash
python split_node_emb.py --dataset mag --emb-file ckpts/ComplEx_mag_0/mag_ComplEx_entity.npy
```



## Training

### ogbn-mag

- Vanilla Seq-HGNN (without Label Propagation & Multi-Stage Training)

  Under this root folder, run

  ```bash
  python -m seq_hgnn.train_amp --lr 0.0005 --num-hidden 512 --num-layers 2 --batch-size 256 --n-batch 250 --dropout 0.5 --num-samples 1800 --dataset ogbn-mag-complex --logsubfix vanilla --epochs 200 --workers 8 --device 0
  ```

- Seq-HGNN + Label Propagation

  ```bash
  python -m seq_hgnn.train_lp --lr 0.0005 --num-hidden 512 --num-layers 2 --batch-size 256 --n-batch 250 --dropout 0.5 --num-samples 1800 --dataset ogbn-mag-complex --logsubfix lp --epochs 200 --workers 8 --device 0
  ```

  

