import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
import pickle
import random
import pandas as pd

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data():
    # print('Loading {} dataset...'.format(dataset))
    G = nx.read_gpickle("graph/graph.gpickle")
    adj = nx.to_numpy_matrix(G)
    adj = sp.coo_matrix(adj, dtype=np.float32)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))
    labels = torch.LongTensor(np.load('masks/label.npy'))
    features = torch.tensor(np.load('masks/ft_mat.npy'), dtype=torch.float32)
    mask = pickle.load( open( "masks/mask.p", "rb" ) )
    random.shuffle(mask)
    print(len(mask))
    mask_train = range(3918)
    val_mask = range(3918, 4905)
    test_mask = range(4905, 6082)
    return adj, features, labels, mask, mask_train,val_mask, test_mask


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


class CustomGraphDataset:
    def __init__(self) -> None:
        self.components = load_data()
        self.data = {}
        self.data['adj'] = self.components[0]
        self.data['features'] = self.components[1]
        self.data['labels']= self.components[2]
        self.data['tweet_mask']= self.components[3]
        self.data['train_mask']= self.components[4]
        self.data['val_mask'] = self.components[5]
        self.data['test_mask'] = self.components[6]
        del self.components

    def get_data(self):
        return self.data

