import torch
from torch_geometric.data import Data
import pandas as pd
import os 
import pickle
import glob
import numpy as np
import sys

data = Data()

## Create Map for tweet_ids and usernames 
tweet_id_to_index = {};username_to_index = {}
df = pd.read_csv(os.getcwd()+'/data/all_data.csv')
tweet_ids = list(df['tweet_id'])
usernames = list(df['username'].unique())

index = 0
for tweet_id in tweet_ids:
    tweet_id_to_index[tweet_id] = index
    index += 1
for usr in usernames:
    username_to_index[usr] = index
    index += 1

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
y = torch.zeros(total_nodes)
for tweet_id in tweet_ids:
    row = df[df['tweet_id']==tweet_id]
    print(row)
    sys.exit(0)