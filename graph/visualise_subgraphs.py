import json
import networkx as nx

fl = json.load(open('graph/parent_comment_train.json','r'))
fl2 = json.load(open('graph/parent_comment_test.json','r'))
fl.update(fl2)
g = nx.Graph()

parent = set()
comment = set()
for key,value in fl.items():
    parent.add(key)
    for u in value:
        comment.add(u)
nodes = list(parent)+list(comment)

for node in list(parent):
    g.add_node(node,color = "green")
for node in list(comment):
    g.add_node(node,color = "blue")

for u,value in fl.items():
    for v in value:
        g.add_edge(u,v)

nx.write_gexf(g,'parent_comment.gexf')
