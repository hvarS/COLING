import torch
from torch_geometric.data import Data
import pandas as pd
import os 
import pickle
import glob
import numpy as np
from visual import create_gexf_graph, create_pydot_viz
import sys

data = Data()

## Create Map for tweet_ids and usernames 
tweet_id_to_index = {};username_to_index = {}
index_to_username = {};index_to_tweet_id = {}
trn_df = pd.read_csv(os.getcwd()+'/data/train.csv')
val_df = pd.read_csv(os.getcwd()+'/data/val.csv')
tst_df = pd.read_csv(os.getcwd()+'/data/test.csv')

df = pd.read_csv(os.getcwd()+'/data/all_data.csv')
df.replace('HOF',1,inplace= True)
df.replace('NONE',0,inplace = True)
tweet_ids = list(df['tweet_id'])
usernames = list(df['user_name'].unique())

index = 0
for tweet_id in tweet_ids:
    tweet_id_to_index[tweet_id] = index
    index_to_tweet_id[index] = tweet_id
    index += 1
tweet_end = index
data.tweet_nodes = range(index)
for usr in usernames:
    username_to_index[usr] = index
    index_to_username[index] = usr
    index += 1
data.usr_nodes = range(tweet_end,index)
data.idx2tweet = index_to_tweet_id
data.idx2usr = index_to_username
data.tweet_ids = tweet_ids
data.usrs = usernames

## Node embeddings 
total_nodes = len(tweet_ids)+len(usernames)
x = torch.zeros(total_nodes,1024)

feature_dict = pickle.load(open('MM/te.pickle','rb'))
usr_embed_location = '../../coling/baseline/user_embed_toxicity/'
trn_loc = usr_embed_location+'train'
val_loc = usr_embed_location+'val'
tst_loc = usr_embed_location+'test'
trn_usr = glob.glob(trn_loc+'/*')
val_usr = glob.glob(val_loc+'/*')
tst_usr = glob.glob(tst_loc+'/*')
usrs = trn_usr+val_usr+tst_usr
for usr in usrs:
    usr_id = usr.split('/')[-1].split('.npy')[0]
    embed = np.mean(np.load(usr),axis = 0)
    feature_dict[usr_id] = embed

for key,value in feature_dict.items():
    if key in usernames:
        x[username_to_index[key]] = torch.tensor(value)
    elif key in tweet_ids:
        x[tweet_id_to_index[key]] = torch.tensor(value)

data.x = x

## Node Labels 
data.num_classes = 2
y = torch.zeros(total_nodes,dtype = torch.long)
torch.fill_(y,-1)
for tweet_id in tweet_ids:
    label = df[df['tweet_id']==tweet_id]['label']
    y[tweet_id_to_index[tweet_id]] = label.iloc[0]
# print(y.unique(return_counts=True))
data.y = y

## Edge Connections 

src = []
dst = []
# Add usr to corresponding tweet connection
for index,row in df.iterrows():
    source = username_to_index[row['user_name']]
    dest = tweet_id_to_index[row['tweet_id']]
    src.append(source)
    dst.append(dest)
 
# Add Comment/Reply to Parent Tweet
chains = list(df['chain_label'].unique())

for chain in chains:
    tweet_chain = df[df['chain_label']==chain]
    parent_tweet_id = tweet_chain.iloc[0,0]
    
    for index,row in tweet_chain.iterrows():
        tweet_id = row['tweet_id']
        if tweet_id != parent_tweet_id:
            source = tweet_id_to_index[tweet_id]
            dest = tweet_id_to_index[parent_tweet_id]
            src.append(source)
            dst.append(dest)

# Add User's Connections with each other 
trn_usr_adj = pickle.load(open('usr_graph/train_matrix.pkl','rb'))
tst_usr_adj = pickle.load(open('usr_graph/test_matrix.pkl','rb'))

for key,value in trn_usr_adj.items():
    try:
        source = username_to_index[key]
        for dest in value:
            dest = username_to_index[dest]
            src.append(source)
            dst.append(dest)
    except:
        pass
data.edge_index = torch.tensor([src,
                                dst])
# print(data.edge_index.shape)

## Creating train, val and test masks
data.trn_mask = torch.zeros(total_nodes,dtype=torch.bool)
data.val_mask = torch.zeros(total_nodes,dtype=torch.bool)
data.tst_mask = torch.zeros(total_nodes,dtype=torch.bool)

for tweet_id in list(trn_df['tweet_id']):
    data.trn_mask[tweet_id_to_index[tweet_id]] = 1
for tweet_id in list(val_df['tweet_id']):
    data.val_mask[tweet_id_to_index[tweet_id]] = 1
for tweet_id in list(tst_df['tweet_id']):
    data.tst_mask[tweet_id_to_index[tweet_id]] = 1

# print(sum(data.tst_mask)) 
# print(data.num_node_features)

#############Visualisation##############
# create_pydot_viz(data)
create_gexf_graph(data)