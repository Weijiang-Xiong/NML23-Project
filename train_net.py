import torch 
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from argparse import ArgumentParser

from model import MODEL_DICT, GCN, GraphTransformer
from dataset import DeezerDataset


def train_node_classifier(model, data, optimizer, criterion, n_epochs=200):

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

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

def parse_arguments(input_args=None):
    
    parser = ArgumentParser()
    
    parser.add_argument('--data-root', type=str, default="./data", help="root folder for data")
    parser.add_argument('--from-raw', type=bool, default=False, action='store_true', help='whether to process from raw features')
    parser.add_argument('--method', type=str, default=None, help='the method to process raw features')
    parser.add_argument('--save-data', type=bool, default=False, action='store_true', help='whether to save the processed data')
    parser.add_argument('--model', type=str, default='gcn', help='type of the model, available: gcn, tf')
    parser.add_argument('--hid-dim', type=int, default=64, help='hidden dimension of the model')
    parser.add_argument('--max-epoch', type=int, default=200, help='number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='weight decay')
    
    args = parser.parse_args(input_args)
    
    return args

if __name__ == "__main__":
    
    args = parse_arguments()
    print("Using these configurations")
    print(args)
    
    ### DEVICE GPU OR CPU : will select GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\nDevice: ", device)
    
    # initialize dataset (only 1 graph in it) and retrieve the data graph
    dataset = DeezerDataset(args.data_root, from_raw=args.from_raw, method=args.method, save_data=args.save_data)
    data_graph = dataset[0].to(device)

    ### DEFINE THE MODEL
    model_class = MODEL_DICT.get(args.model, GCN)
    print("Using model class {}".format(model_class.__name__))
    
    model = model_class(input_size=dataset.num_node_features, 
                             output_size=dataset.num_classes,
                             hidden_size=args.hid_dim).to(device)
    print("Detailed Model Definition:")
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=args.learning_rate, 
                                 weight_decay=args.weight_decay)
    
    criterion = nn.CrossEntropyLoss()
    gcn = train_node_classifier(model, data_graph, optimizer, criterion, n_epochs=args.max_epoch)

    test_acc = eval_node_classifier(model, data_graph, data_graph.test_mask)
    
    print(f'Test Acc: {test_acc:.3f}')