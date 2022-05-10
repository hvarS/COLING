# from dataset import ToxicTweetDataset
# import torch 
# import random 
# from torch_geometric.data import HeteroData
# from torch_geometric.nn import to_hetero,GraphConv
# # data = ToxicTweetDataset('','')


# data = HeteroData()

# # 100 Papers, 10 Authors 
# # 768 features per paper, 1 feature per author 
# # 199 citations
# data['paper'].x = torch.randn(100,768)
# data['author'].x = torch.randn(10,1)

# hetero_edge = torch.zeros((2,150))

# for i in range(2):
#     for j in range(150):
#         if i == 0:
#             hetero_edge[i][j] = random.randint(1,10)
#         else:
#             hetero_edge[i][j] = random.randint(1,100)

# data['paper','cites','paper'].edge_index = torch.randint(1,100,(2,200))
# data['author','writes','paper'].edge_index = hetero_edge
# data['author','knows','author'].edge_index = torch.randint(1,10,(2,25))

# node_types, edge_types = data.metadata()

# class GNN(torch.nn.Module):
#     def __init__(self, hidden_channels, out_channels):
#         super().__init__()
#         self.conv1 = GraphConv((-1, -1), hidden_channels)
#         self.conv2 = GraphConv((-1, -1), out_channels)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index).relu()
#         x = self.conv2(x, edge_index)
#         return x

# # # print(data.metadata())
# model = GNN(hidden_channels=768,out_channels=1)
# model = to_hetero(model, data.metadata())
# # print(data.stores)
# # print(data.node_types)

# out = model(data.x_dict,data.edge_index_dict)
# print(out.shape)

import pandas as pd
df = pd.read_csv('Full_train.csv')
df = df.iloc[:,5:]
df.to_csv('Full_train.csv',index=False)