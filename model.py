import torch 
import torch.nn as nn
import torch_geometric.nn as graphnn
import torch.nn.functional as F

class GCN(nn.Module):
    """ A baseline GCN model from the exercises
    """
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
    
class GraphTransformer(nn.Module):
    """ Another baseline model that replaces the GCN conv with transformer conv
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.graphconv1 = graphnn.TransformerConv(input_size, hidden_size)
        self.graphconv2 = graphnn.TransformerConv(hidden_size, hidden_size)
        self.graphconv3 = graphnn.TransformerConv(hidden_size, output_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.graphconv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.graphconv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.graphconv3(x, edge_index)
        
        return F.log_softmax(x, dim=1)
    
MODEL_DICT = {
    "gcn": GCN,
    "tf": GraphTransformer
}
