
from typing import Callable, Optional

import torch
import numpy as np 

from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import DeezerEurope
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.data import Data, InMemoryDataset, download_url

class DeezerDataset(InMemoryDataset):
    
    """ modified from torch_geometric.datasets.DeezerEurope
    """
    
    url = 'https://graphmining.ai/datasets/ptg/deezer_europe.npz'

    def __init__(self, root, transform=None, pre_transform=None, val_size=0.1, test_size=0.2):
        
        super().__init__(root, transform, pre_transform)
        
        data, slices = torch.load(self.processed_paths[0])
        data = RandomNodeSplit(num_val=val_size, num_test=test_size)(data)
        
        self.data, self.slices = data, slices

    @property
    def raw_file_names(self) -> str:
        return 'deezer_europe.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        data = np.load(self.raw_paths[0], 'r', allow_pickle=True)
        x = torch.from_numpy(data['features']).to(torch.float)
        y = torch.from_numpy(data['target']).to(torch.long)
        edge_index = torch.from_numpy(data['edges']).to(torch.long)
        edge_index = edge_index.t().contiguous()

        data = Data(x=x, y=y, edge_index=edge_index)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
        