import torch 
import torch_geometric.nn as gnn
from torch_geometric.data import HeteroData
import random 
from torch_geometric.nn import GraphConv, to_hetero
import glob
import pandas as pd
import pickle

class ToxicTweetDataset(object):
    def __init__(self,dataset_path,matrix_path) -> None:
        super().__init__()
        self.mp = matrix_path
        self.dp = dataset_path
        self.get_csv()
        self.get_matrix()
        
        self.usrnmToIndex = {}
        self.generate_username_to_index()

        self.train_user_adj = torch.zeros(self.len_users,self.len_users)
        self.generate_adj()
        
    def get_csv(self):
        files = glob.glob(self.dp+'*.csv')
        for file in files:
            if 'train' in file:
                self.train_data = pd.read_csv(file)
            else:
                self.test_data = pd.read_csv(file)
    
    def get_matrix(self):
        files = glob.glob(self.mp+'*.pkl')
        for file in files:
            if 'train' in file:
                self.train_user_matrix_dt = pickle.load(open(file,'rb'))
            else:
                self.test_user_matrix_dt = pickle.load(open(file,'rb'))
    
    def generate_username_to_index(self):
        users = list(self.train_data['username'].unique())+list(self.test_data['username'].unique())
        self.len_users = len(users)
        self.usrnmToIndex = {usrnm:i for i,usrnm in enumerate(users)}
    
    def generate_adj(self):
        frm = []
        to = []
        for src,dests in self.train_user_matrix_dt.items():
            src = self.usrnmToIndex[src]
            for dest in dests:
                dest = self.usrnmToIndex[dest]
                self.train_user_adj[src][dest] = 1
                frm.append(src)
                to.append(dest)
        self.train_usr_edge_index = torch.stack([torch.tensor(frm),torch.tensor(to)])

        # print(self.train_edge_index.shape)
        frm = []
        to = []
        for src,dests in self.test_user_matrix_dt.items():
            src = self.usrnmToIndex[src]
            for dest in dests:
                dest = self.usrnmToIndex[dest]
                self.train_user_adj[src][dest] = 1
                frm.append(src)
                to.append(dest)
        self.test_usr_edge_index = torch.stack([torch.tensor(frm),torch.tensor(to)])

        # print(self.test_edge_index.shape)
            

