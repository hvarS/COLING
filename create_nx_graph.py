import numpy as np 
import pandas as pd 
import networkx as nx
import os
import pickle 
import json

# trn_df = pd.read_csv(os.getcwd()+'/data/train.csv')
# tst_df = pd.read_csv(os.getcwd()+'/data/test.csv')
# val_df = pd.read_csv(os.getcwd()+'/data/val.csv')
# df = pd.concat([trn_df,val_df,tst_df],ignore_index=True)
# df.to_csv('data/all_data.csv',index = False)
df = pd.read_csv(os.getcwd()+'/data/all_data.csv')
G = nx.Graph() 

# Add the type of nodes that we'll add 
for index,row in df.iterrows():
    G.add_nodes_from([row['tweet_id'],row['user_name']])

# Add usr to corresponding tweet connection
for index,row in df.iterrows():
    G.add_edge(row['user_name'],row['tweet_id'])
# Add Comment/Reply to Parent Tweet
# chains = list(df['chain_label'].unique())

# for chain in chains:
#     tweet_chain = df[df['chain_label']==chain]
#     parent_tweet_id = tweet_chain.iloc[0,0]
    
#     for index,row in tweet_chain.iterrows():
#         tweet_id = row['tweet_id']
#         if tweet_id != parent_tweet_id:
#             G.add_edge(tweet_id,parent_tweet_id)

# print(list(G.nodes())[:100])
# Add User's Connections with each other 
trn_usr_adj = pickle.load(open('graph/usr_graph/train_matrix.pkl','rb'))
tst_usr_adj = pickle.load(open('graph/usr_graph/test_matrix.pkl','rb'))

for key,value in trn_usr_adj.items():
    src = key
    for dest in value:
        G.add_edge(src,dest)

# Add parent Comment connections
fl = json.load(open('graph/tweet_graph/parent_comment_train.json','r'))
fl2 = json.load(open('graph/tweet_graph/parent_comment_test.json','r'))
fl.update(fl2)
for u,value in fl.items():
    for v in value:
        G.add_edge(u,v)

# Add Comment Reply Connections 
fl = json.load(open('graph/tweet_graph/comment_reply_train.json','r'))
fl2 = json.load(open('graph/tweet_graph/comment_reply_test.json','r'))
fl.update(fl2)
for u,value in fl.items():
    for v in value:
        G.add_edge(u,v)

# Add reply reply connections 
fl = json.load(open('graph/tweet_graph/reply_reply_train.json','r'))
fl2 = json.load(open('graph/tweet_graph/reply_reply_test.json','r'))
fl.update(fl2)
for u,value in fl.items():
    for v in value:
        G.add_edge(u,v)

print(G)

newDir = os.path.join(os.getcwd(),'graph')
if not os.path.exists(newDir):
    os.mkdir(newDir)

nx.write_gpickle(G, os.path.join(newDir,'graph'+".gpickle"))
