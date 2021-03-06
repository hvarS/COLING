from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from dataset import data


print()
print('======================')
print(f'Number of features: {data.num_features}')
print(f'Number of classes: {data.num_classes}')


print()
print(data)
print('===========================================================================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.trn_mask.sum()}')
print(f'Training node label rate: {int(data.trn_mask.sum()) / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')