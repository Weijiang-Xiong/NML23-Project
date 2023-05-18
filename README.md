# NML23-Project
Course project for EE-452 Networked Machine Learning at EPFL, the task is node classification on Deezer Europe dataset.


## Installation

This project is built upon `pytorch` and `pytorch_geometric`, we assume the existence of a conda environment called pytorch

```
conda activate pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg
```
## Project Outline

We use Adam as optimizer and cross entropy loss. 

## Example Usage

The following command will train a GCN model with hidden dimension 64 for 200 epochs/iterations, and the learning rate and weight decay in the optimizer are set to 0.0001 and 0.001 respectively. 

```
python train_net.py --model gcn --hid-dim 64 --max-epoch 200 --learning-rate 0.0001 --weight-decay 0.001
```

## References
A blog post about using pyg https://towardsdatascience.com/graph-neural-networks-with-pyg-on-node-classification-link-prediction-and-anomaly-detection-14aa38fe1275

https://pytorch-geometric.readthedocs.io/en/latest/ 
