import os
from turtle import fillcolor
import torch
# Helper function for visualization.
import matplotlib.pyplot as plt
import pydot 
from sklearn.manifold import TSNE

def visualize(h, color, location):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.savefig(location) 


def create_pydot_viz(data):
    graph = pydot.Dot('mygraph',graph_type = "graph")
    for node in range(data.num_nodes):
        if node in data.usr_nodes:
            u = pydot.Node(name=str(data.idx2usr[node]),label=str(node),style = "filled", fillcolor="gold")
            graph.add_node(u)
        else:
            u = pydot.Node(name=str(data.idx2tweet[node]),label=str(node),style = "filled", fillcolor="blue")
            graph.add_node(u)
    for u,v in zip(data.edge_index[0].numpy(),data.edge_index[1].numpy()):
        edge = pydot.Edge(str(u),str(v))
        graph.add_edge(edge)
    graph.write_png("graph_visual.png")