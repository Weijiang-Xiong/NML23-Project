import os
import json
import zipfile
from itertools import chain

import torch
import numpy as np 
import pandas as pd 
import networkx as nx 

from scipy.sparse import coo_matrix
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.data import Data, InMemoryDataset, download_url
from karateclub.node_embedding import FeatherNode
from sklearn.decomposition import TruncatedSVD
from node2vec import Node2Vec

class DeezerDataset(InMemoryDataset):
    
    """ 
        This class implements various preprocessing methods for Deezer Europe dataset (paper: https://arxiv.org/abs/2005.07959). 
    
        This dataset contains a single graph whose nodes are deezer users from european countries. The edges are mutual following relationship between users, and the node features are the artists (~30k in total) liked by the users.
        The nodes are also labeled with the gender of the user. 
        
        num_nodes: 28281, num_edges: 185504 (a mutual following relation makes two 1's in a symmetric adjacency matrix)
        
        This implementation is modified from torch_geometric.datasets.DeezerEurope, and the key differences are the feature processing methods implemented in `self.process_raw_feature`, i.e., how to embed the liked artists of each user. Whereas the nodes, edges and labels are kept as they are. 
        
    """
    # this URL provides the original data where the node features are the artists liked by a user
    url_raw_features = 'https://snap.stanford.edu/data/deezer_europe.zip'
    # this version provides 128d processed node features
    url = 'https://graphmining.ai/datasets/ptg/deezer_europe.npz'

    def __init__(self, root, transform=None, pre_transform=None, val_size=0.1, test_size=0.2,
                 from_raw=False, method="feather", save_data=False):
        """
        Args:
            `root` (str): root folder to store datasets
            
            `val_size` (float, optional): size of validation set. Defaults to 0.1.
            
            `test_size` (float, optional): size of test set. Defaults to 0.2.
            
            `from_raw_feature` (bool, optional): whether to use the original features (artists liked by the users). Defaults to True. if set to false, will use the 128-D preprocessed node feature from torch geometric (yet the processing method is not found in its document), and `method` will not be effective. 
            
            `raw_process_method` (str, optional): methods to use when processing raw features (`from_raw` should be `True`). Defaults to `"feather"`, which is the same as the dataset paper. 
            
            `save_data`: whether to save the pre-processed data for future use. Recommended when the data dimension is properly reduced (i.e. with SVD), otherwise the feature file will be too big to save. 
        
        
        All possible choices for `method`, when `from_raw` is set to `True`: 
        
            `None`: the node features will be kept unprocessed, and the data graph will be returned in networkx format. 
            
            `"feather"`: the attribute embedding methods, feature dimension ~30k
            
            `"feather+svd"`: use SVD to reduce the dimension of feather features to 128, aligned with the preset features.
            
            `"binary"`: use a 0/1 vector of length num_artists to represent the liked artists of each user, a 1 in the vector means the user likes the corresponding artist. 
            
            `"svd"`: use SVD to reduce the dimension of the `binary` features down to 128, keeping aligned with the preset features.
            
            `"node2vec+svd"`: use node2vec to encode the structure of the graph, and use svd of the `binary` features to represent feature information, both have 64-D and are concatenated to 128-D features. 
            
            `"binary+n2v"`: concatenate 128-D node2vec features to the `binary` features. 
            
            
        After initializing the dataset, use dataset[0] to retrieve the data graph in torch_geometric.data.Data format, since this dataset contains only one graph. 
        
        All configurations are listed under `if __name__=='__main__`. 
        
        """
        # super class init will rely on this flag
        self.from_raw = from_raw
        self.method = method
        self.save_data = save_data
        
        super().__init__(root, transform, pre_transform)
        
        if os.path.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.data, self.slices = self._processed_data, self.processed_slices
                
        if isinstance(self._data, Data):
            self.data = RandomNodeSplit(num_val=val_size, num_test=test_size)(self._data)


    @property
    def raw_file_names(self) -> str:
        return 'deezer_europe.zip' if self.from_raw else 'deezer_europe.npz'

    @property
    def processed_file_names(self) -> str:
        
        if not self.from_raw:
            return 'data_preset.pt'
        elif self.method is None:
            return 'data_nx_graph.pt'   
        else:
            return 'data_{}.pt'.format(self.method)
    
    # @property
    # def processed_dir(self) -> str:
    #     suffix = "_from_raw" if self.from_raw_feature else "_from_preset"
    #     return super().processed_dir + suffix
    
    def download(self):
        url = self.url_raw_features if self.from_raw else self.url
        download_url(url, self.raw_dir)

    def process(self):
        if self.from_raw:
            data = self.process_raw_feature()
        else:
            data = self.process_preset_feature()
        
        self.post_processing(data)

    def post_processing(self, data):
        
        if self.pre_transform is not None and isinstance(data, Data):
            data = self.pre_transform(data)

        if self.save_data:
            print("Saving the preprocessed data to {}".format(self.processed_paths[0]))
            torch.save(self.collate([data]), self.processed_paths[0])
        else:
            print("Keeping the data in memory, linked to self._processed_data")
            self._processed_data, self.processed_slices = self.collate([data])

    def process_preset_feature(self):
        print("Using the provided dataset from PyTorch Geometric")
        data = np.load(self.raw_paths[0], 'r', allow_pickle=True)
        # data['features'].shape: (28281, 128)
        x = torch.from_numpy(data['features']).to(torch.float)
        # data['features'].shape (28281,)
        y = torch.from_numpy(data['target']).to(torch.long)
        # data['edges'] shape (185504, 2), format ([node1, node2], ...)
        edge_index = torch.from_numpy(data['edges']).to(torch.long)
        edge_index = edge_index.t().contiguous()
        # edge_index has shape (2, 185504), transposed from data['edge']
        data = Data(x=x, y=y, edge_index=edge_index)
        
        return data 

    @staticmethod
    def reduce_data_dim(X, reduction_dimensions=128, svd_iterations=20, seed=42):
        """ 
            Use SVD to reduce data dimension. This part is copied from FeatherNode._reduce_dimensions, and modified to a static method
            
            Input: X of shape (num_nodes, dim)
            Output: X of shape (num_nodes, reduction_dimensions)
            
        """
        svd = TruncatedSVD(
            n_components=reduction_dimensions,
            n_iter=svd_iterations,
            random_state=seed,
        )
        svd.fit(X)
        X = svd.transform(X)
        return X
        
    def process_raw_feature(self):
        
        raw_file_path = "{}/{}".format(self.raw_dir, self.raw_file_names)
        with zipfile.ZipFile(raw_file_path, 'r') as zip_ref:
            zip_ref.extractall(self.raw_dir)
        
        file_names = {'edge': 'deezer_europe_edges.csv', 
                      'feature': 'deezer_europe_features.json', 
                      'target': 'deezer_europe_target.csv'}
        
        file_paths = {k:"{}/deezer_europe/{}".format(self.raw_dir, v) for k, v in file_names.items()}
        
        ####################################################
        ############### processing edges  ##################
        ####################################################
        
        # note the edges in the raw data is not symmetric, so we need to make it symmetric 
        edge = pd.read_csv(filepath_or_buffer=file_paths['edge'], header=0).to_numpy()
        # if [node1, node2] exists in the raw data, we add [node2, node1] to the edge lists
        sym_edge = np.concatenate(
            (
            edge.copy()[:, 1].reshape(-1, 1),
            edge.copy()[:, 0].reshape(-1, 1)
            ), 
            axis=1)
        edge_index = np.concatenate((edge, sym_edge), axis=0)
        edge_index = torch.from_numpy(edge_index).to(torch.long).t().contiguous()
        
        ####################################################
        ############### processing labels ##################
        ####################################################
        
        # each row in the raw target file is in format [node_id, label (0/1)]
        target = pd.read_csv(filepath_or_buffer=file_paths['target'], header=0).to_numpy()
        node_ids, labels = target[:, 0], target[:, 1]
        # assert node ids should be contiguous (i.e. from 0 to num_node-1)
        assert np.all(np.diff(node_ids)==1) and node_ids[0]==0 and node_ids[-1]==len(node_ids)-1
        labels = torch.from_numpy(labels).to(torch.long)
        
        ####################################################
        ############### processing features  ###############
        ####################################################
        
        with open(file_paths['feature']) as feature_file:
            user_features = json.load(feature_file)

        # Extract artist IDs
        artists = set(chain.from_iterable(user_features.values()))

        # Create feature matrix
        num_users, num_artists = len(user_features), max(artists) + 1
        like_relation, user_idx, artist_idx = [[] for _ in range(3)]

        # 1 if the artist is liked by the user, 0 otherwise
        for user, liked_artists in user_features.items():
            for artist in liked_artists:
                like_relation.append(1)
                user_idx.append(int(user))
                artist_idx.append(int(artist))

        feature_matrix_sparse = coo_matrix(
            (
            np.array(like_relation), 
            (np.array(user_idx), np.array(artist_idx))
            ), 
            shape=(num_users, num_artists))

        
        ####################################################
        ############### data container 1 ###################
        ####################################################
        
        # construct the data graph in networkx format
        graph = nx.Graph()
        graph.add_nodes_from(node_ids)
        graph.add_edges_from(edge_index.T.numpy()) # networkx requires shape (num_edge, 2)
        
        ####################################################
        ############### node embeddings ####################
        ####################################################
        
        node_embeddings = None
        
        if self.method == 'feather':
            # according to section 4.2 the feature dimension is reduced to 128 by svd
            # other parameters are kept default
            print("Embedding raw node features with feather method, takes time ...")
            embed_method = FeatherNode(reduction_dimensions=128)
            embed_method.fit(graph, feature_matrix_sparse)
            node_embeddings = embed_method.get_embedding()
            
        elif self.method == 'feather+svd':
            
            print("Embedding raw node features with feather method, takes time ...")
            embed_method = FeatherNode(reduction_dimensions=128)
            embed_method.fit(graph, feature_matrix_sparse)
            node_embeddings = embed_method.get_embedding()
            
            print("Reducing the data dimension with SVD, takes time ...")
            node_embeddings = self.reduce_data_dim(node_embeddings, reduction_dimensions=128)
        
        elif self.method == 'node2vec+svd':
            
            # to keep the output embeddings in 128 dim, align with feather+svd
            print("Embedding graph structure with Node2Vec, takes time ...")
            embed_method = Node2Vec(graph, dimensions=64, workers=1)
            model = embed_method.fit()
            structural_embeddings = np.array([model.wv[n] for n in graph.nodes()])
            
            print("Reducing binary node feature with SVD, also takes time")
            attribute_embeddings = self.reduce_data_dim(feature_matrix_sparse, reduction_dimensions=64)
            node_embeddings = np.concatenate([structural_embeddings, attribute_embeddings], axis=1)
            
        elif self.method == 'binary':
            
            print("Using the binary features vectors as node embeddings")
            node_embeddings = feature_matrix_sparse.toarray()
            
        elif self.method == 'svd':
            
            print("Using reduced binary feature vectors as node embeddings")
            # using a sparse matrix is much faster than np array
            node_embeddings = self.reduce_data_dim(feature_matrix_sparse, reduction_dimensions=128)
        
        elif self.method == "binary+n2v":
            
            print("Embedding graph structure with Node2Vec, takes time ...")
            embed_method = Node2Vec(graph, dimensions=128, workers=1)
            model = embed_method.fit()
            structural_embeddings = np.array([model.wv[n] for n in graph.nodes()])
            
            attribute_embeddings = feature_matrix_sparse.toarray()
            node_embeddings = np.concatenate([structural_embeddings, attribute_embeddings], axis=1)
            
        elif self.method == "n2v":
            # this is a baseline case, only node2vec embeddings are used
            print("Embedding graph structure with Node2Vec, takes time ...")
            embed_method = Node2Vec(graph, dimensions=128, workers=1)
            model = embed_method.fit()
            node_embeddings = np.array([model.wv[n] for n in graph.nodes()])
        
        # add the raw features directly to the graph, in this case we can't make it into Data class
        # because the features have varying length, and can not be packed to a tensor.
        else:
            node_feat_and_label = {nid:{"label":labels[nid], 'features':user_features[str(nid)]} 
                                   for nid in node_ids}
            nx.set_node_attributes(G=graph, values=node_feat_and_label)

        ####################################################
        ############### data container 2 ###################
        ####################################################
        
        if node_embeddings is None:
            data = graph
        else:
            data = Data(x=torch.from_numpy(node_embeddings).to(torch.float32), 
                        y=labels, 
                        edge_index=edge_index)
        
        return data 
    

if __name__ == "__main__":
    
    DeezerDataset("./data/", from_raw=False, save_data=True)
    DeezerDataset("./data/", from_raw=True, method=None, save_data=True) # returns nx graph, not for network training
    DeezerDataset("./data/", from_raw=True, method='feather', save_data=False)
    DeezerDataset("./data/", from_raw=True, method='feather+svd', save_data=True)
    DeezerDataset("./data/", from_raw=True, method='node2vec+svd', save_data=True)
    DeezerDataset("./data/", from_raw=True, method='binary', save_data=False)
    DeezerDataset("./data/", from_raw=True, method='svd', save_data=True)
    DeezerDataset("./data/", from_raw=True, method='binary+n2v', save_data=False)