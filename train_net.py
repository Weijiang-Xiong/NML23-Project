import pickle

import torch 
import torch.nn as nn
torch.manual_seed(42)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

from sklearn.metrics import f1_score
from argparse import ArgumentParser

from model import MODEL_DICT, GCN, GraphTransformer
from dataset import DeezerDataset


def train_node_classifier(model, data, optimizer, criterion, n_epochs=200, 
                          plot=True, save_note=None):

    record = []
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        val_acc = eval_node_classifier(model, data, data.val_mask)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val Acc: {val_acc:.3f}')

        record.append((epoch, loss.item(), val_acc))
        
    test_acc = eval_node_classifier(model, data_graph, data_graph.test_mask)
    print("Test Accuracy is {:.3f}".format(test_acc))
    
    if plot:
        record = np.array(record)
        xs, losses, accs = record[:, 0], record[:, 1], record[:, 2]

        fig, ax = plt.subplots(1,2, figsize=(12, 5))
        
        ax[0].plot(xs, losses, label="Train Loss")
        ax[0].set_xlabel("Epochs")
        ax[0].set_title("Training Loss")

        ax[1].plot(xs, accs, label="Val. Acc.", color="#1f77b4")
        ax[1].axhline(y=test_acc, linestyle="--", label="Test Acc.", color="#ff7f0e")
        # add text for the test accuracy line 
        ax[1].text(xs[int(0.1*len(xs))], test_acc, "{:.3f}".format(test_acc), fontsize=12, color="#ff7f0e")
        
        ax[1].set_xlabel("Epochs")
        ax[1].set_title("Accuracy")

        ax[0].legend(loc="upper right", bbox_to_anchor=(1, 1))
        ax[1].legend(loc="lower right", bbox_to_anchor=(1, 0))

        save_name = "training_log_{}".format(save_note if save_note else "None")
        fig.savefig("figs/{}.png".format(save_name), dpi=300)
        with open("figs/{}.pkl".format(save_name), "wb") as f:
            pickle.dump({'record':record, 'test_acc':test_acc}, f)
            
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
    parser.add_argument('--from-raw', default=False, action='store_true', help='whether to process from raw features')
    parser.add_argument('--method', type=str, default=None, help='the method to process raw features')
    parser.add_argument('--save-data', default=False, action='store_true', help='whether to save the processed data')
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
    
    if args.from_raw:
        assert args.method is not None, "Raw features must have a processing method"
        if args.save_data:
            assert "svd" in args.method, "Can't save data without dimension reduction, the file will be too big"
        
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
    model = train_node_classifier(model, data_graph, optimizer, criterion, n_epochs=args.max_epoch,
                                  plot=True, save_note=args.method)