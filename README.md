## Project Outline

This repo contains the codes for Xinyu Liu and Weijiang Xiong's course project for EE-452 Networked Machine Learning at EPFL, the task is node classification on Deezer Europe dataset.

In this project we investigate the user graph of Deezer Europe dataset. The nodes are users, the edges are mutual following relationship, and the node features are the liked artists of a user. We explore the graph and train neural networks with various feature embeddings to classify the gender of the users. Please refer to the notebook `Presentation.ipynb` for a quick tour.

The paper of the dataset propose the [FEATHER](https://arxiv.org/abs/2005.07959) method to learn node embedding from attributed graph.
Following the authors' codes, we use [Karate Club](https://github.com/benedekrozemberczki/karateclub) to process raw node features.

The slides and project report can be found in `./doc` folder

## Installation

This project is built upon `pytorch` and `pytorch_geometric`, we assume the existence of a conda environment called pytorch

```
conda activate pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg
```

Feature processing of the raw dataset needs [Karate Club](https://github.com/benedekrozemberczki/karateclub)

```
python -m pip install karateclub
```

## Example Usage

The following command will train a GCN model with hidden dimension 64 for 200 epochs/iterations, and the learning rate and weight decay in the optimizer are set to 0.0001 and 0.001 respectively.

```
python train_net.py --model gcn --hid-dim 64 --max-epoch 200 --learning-rate 0.0001 --weight-decay 0.001
```

All the commands used in this project are listed in `run_everything.py` and one can simply run it. However, please remember the experiments with `binary` or `feather` features may take up to 8 GB of GPU memory because the embedding matrix is very big. Please also be aware that computing node2vec features can take even half an hour, because the graph is too big.

```bash
python run_everything.py
```

## References

A blog post about using pyg https://towardsdatascience.com/graph-neural-networks-with-pyg-on-node-classification-link-prediction-and-anomaly-detection-14aa38fe1275

https://pytorch-geometric.readthedocs.io/en/latest/
