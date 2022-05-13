import os
from turtle import fillcolor
import torch
# Helper function for visualization.
import matplotlib.pyplot as plt
import pydot 
import networkx as nx 
from sklearn.manifold import TSNE
from pyvis.network import Network
from torch_geometric.utils import to_networkx

def visualize(h, color, location):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.savefig(location) 


def create_nx_graph(data):
    nx_Graph = to_networkx(data)
    nx.draw(nx_Graph)
    plt.savefig("graph.png")

def create_gexf_graph(data):
    nx_Graph = to_networkx(data)
    nx.write_gexf(nx_Graph,'graph.gexf')

def create_pydot_viz(data):
    # graph = pydot.Dot('mygraph',graph_type = "graph")
    # for node in range(data.num_nodes):
    #     if node in data.usr_nodes:
    #         u = pydot.Node(name=str(data.idx2usr[node]),label=str(node),style = "filled", fillcolor="gold")
    #         graph.add_node(u)
    #     else:
    #         u = pydot.Node(name=str(data.idx2tweet[node]),label=str(node),style = "filled", fillcolor="blue")
    #         graph.add_node(u)
    # for u,v in zip(data.edge_index[0].numpy(),data.edge_index[1].numpy()):
    #     edge = pydot.Edge(str(u),str(v))
    #     graph.add_edge(edge)
    pass
    

def create_pyvis_graph(data):
    net = Network()
    for node in range(data.num_nodes):
        if node in data.usr_nodes:
            # u = pydot.Node(name=str(data.idx2usr[node]),label=str(node),style = "filled", fillcolor="gold")
            net.add_node(node,label=str(data.idx2usr[node]),colot = "blue")
        else:
            # u = pydot.Node(name=str(data.idx2tweet[node]),label=str(node),style = "filled", fillcolor="blue")
            net.add_node(node,label=str(data.idx2tweet[node]),color = "yellow")
    for u,v in zip(data.edge_index[0].numpy(),data.edge_index[1].numpy()):
        # edge = pydot.Edge(str(u),str(v))
        net.add_edge(u,v)
    net.show('graph.html')