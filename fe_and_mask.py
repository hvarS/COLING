import networkx as nx 
import pickle
import os
import numpy as np 
import pandas as pd
import sys
from tqdm import tqdm

#Read Data 
df = pd.read_csv('data/all_data.csv')
tweet_ids = list(df['tweet_id'].unique())
usernames = list(df['username'].unique())

# # Random features 
# feature_dict = {}
# for usr in usernames:
#     feature_dict[usr] = np.random.randn(768)
# for tweet_id in tweet_ids:
#     feature_dict[tweet_id] = np.random.randn(768)

#Actual Features
feature_dict = pickle.load(open('MM/te.pickle','rb'))
for usr in usernames:
    feature_dict[usr] = np.random.randn(1024)


# Get labels
binarise = lambda x: 1 if x=='HOF' else 0
labList = list(df['new_label'].apply(binarise))
lab = {tid:label for tid,label in zip(tweet_ids,labList)}

# Masking 
i = 0
masker = []
labels = []
G = nx.read_gpickle("data/graph.gpickle")
ft_mat = []
for node in tqdm(list(G.nodes())):
    if node in tweet_ids:
        masker.append(i)
        labels.append(lab[node])
    ft_mat.append(feature_dict[node].reshape(1024))
    i = i+1

newDir = os.path.join(os.getcwd(),'MM')
if not os.path.exists(newDir):
    os.mkdir(newDir)

pickle.dump(masker, open( os.path.join(newDir,"mask.p"), "wb" ))
ft_mat = np.array(ft_mat)
labels = np.array(labels)
np.save(os.path.join(newDir,"label.npy"), labels)
np.save(os.path.join(newDir,"ft_mat.npy"), ft_mat)