# %% [markdown]
# A blog post about using pyg https://towardsdatascience.com/graph-neural-networks-with-pyg-on-node-classification-link-prediction-and-anomaly-detection-14aa38fe1275

# %%

########## INSTALL TORCH GEOMETRIC ##################
# https://pytorch-geometric.readthedocs.io/en/latest/ 
#####################################################
# import torch 

# def format_pytorch_version(version):
#   return version.split('+')[0]

# TORCH_version = torch.__version__
# TORCH = format_pytorch_version(TORCH_version)

# def format_cuda_version(version):
#   return 'cu' + version.replace('.', '')

# CUDA_version = torch.version.cuda
# CUDA = format_cuda_version(CUDA_version)

# !pip install torch-scatter     -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
# !pip install torch-sparse      -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
# !pip install torch-cluster     -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
# !pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
# !pip install torch-geometric

# %%
#####################################################
################## PACKAGES #########################
#####################################################
import numpy as np
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
import torch_geometric.nn as graphnn
from sklearn.metrics import f1_score
from torch_geometric.datasets import DeezerEurope
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.transforms import RandomNodeSplit

# %%
# custom dataset
class DeezerDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DeezerDataset, self).__init__(root, transform, pre_transform)

        # Load the DeezerEurope dataset
        dataset = DeezerEurope(root=self.root)
        data = dataset[0]

        data.num_classes = dataset.num_classes
        data.num_node_features = dataset.num_node_features

        # Randomly split the nodes into train and test sets
        # Use 70% of nodes for training, 20 for test, 10% for validation
        data = RandomNodeSplit(num_val=0.1, num_test=0.2)(data)

        self.data, self.slices = self.collate([data])

# %%
dataset = DeezerDataset('.')
data = dataset[0]

# %%
raw_data = data = np.load('raw/deezer_europe.npz', 'r', allow_pickle=True)
raw_data['features'].shape 

# %%
class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.graphconv1 = graphnn.GCNConv(input_size, hidden_size)
        self.graphconv2 = graphnn.GCNConv(hidden_size, hidden_size)
        self.graphconv3 = graphnn.GCNConv(hidden_size, output_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.graphconv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.graphconv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.graphconv3(x, edge_index)
        
        return F.log_softmax(x, dim=1)

        # return x

# %%
def train_node_classifier(model, data, optimizer, criterion, n_epochs=200):

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        pred = out.argmax(dim=1)
        acc = eval_node_classifier(model, data, data.val_mask)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val Acc: {acc:.3f}')

    return model

def eval_node_classifier(model, data, mask):

    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[mask] == data.y[mask]).sum()
    acc = int(correct) / int(mask.sum())

    return acc

# %%
### DEVICE GPU OR CPU : will select GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("\nDevice: ", device)

### Max number of epochs
max_epochs = 200

### DEFINE THE MODEL
model = GCN(input_size=data.num_node_features, hidden_size=64, output_size=data.num_classes).to(device)
data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=10e-4)
criterion = nn.CrossEntropyLoss()
gcn = train_node_classifier(model, data, optimizer, criterion)

test_acc = eval_node_classifier(model, data, data.test_mask)
print(f'Test Acc: {test_acc:.3f}')

# %%



