import networkx as nx 
import pickle
import os
import numpy as np 
import pandas as pd
import sys
from tqdm import tqdm
import glob

#Read Data 
G = nx.read_gpickle("graph/graph.gpickle")
df = pd.read_csv('data/all_data.csv')
tweet_ids = list(df['tweet_id'].unique())
usernames = list(df['user_name'].unique())


# Random features 
# feature_dict = {}
# for usr in usernames:
#     feature_dict[usr] = np.random.randn(1024)
# for tweet_id in tweet_ids:
#     feature_dict[tweet_id] = np.random.randn(1024)

# Actual Features
feature_dict = pickle.load(open('embeddings/tweet_embeddings.pickle','rb'))

usr_embed_location = 'embeddings/'
trn_loc = usr_embed_location+'train_users_mean.pickle'
val_loc = usr_embed_location+'val_users_mean.pickle'
tst_loc = usr_embed_location+'test_users_mean.pickle'

trn_usr_embed = pickle.load(open(trn_loc,'rb'))
val_usr_embed = pickle.load(open(val_loc,'rb'))
tst_usr_embed = pickle.load(open(tst_loc,'rb'))
usr_embed = {}
usr_embed.update(trn_usr_embed)
usr_embed.update(val_usr_embed)
usr_embed.update(tst_usr_embed)

usr_embed = {k:v.cpu().numpy() for k,v in usr_embed.items()}
feature_dict.update(usr_embed)

# trn_usr = glob.glob(trn_loc+'/*')
# val_usr = glob.glob(val_loc+'/*')
# tst_usr = glob.glob(tst_loc+'/*')
# usrs = trn_usr+val_usr+tst_usr
# for usr in usrs:
#     usr_id = usr.split('/')[-1].split('.npy')[0]
#     embed = np.mean(np.load(usr),axis = 0)
#     # print(embed.shape)
#     feature_dict[usr_id] = embed
#     # print(usr_id in list(G.nodes()))
#     # sys.exit(0)

# Get labels
binarise = lambda x: 1 if x=='HOF' else 0
labList = list(df['label'].apply(binarise))
lab = {tid:label for tid,label in zip(tweet_ids,labList)}

# Masking 
i = 0
masker = []
labels = []
ft_mat = []
for node in tqdm(list(G.nodes())):
    if node in tweet_ids:
        masker.append(i)
        labels.append(lab[node])
    try:
        ft_mat.append(feature_dict[node].reshape(1024))
    except:
        ft_mat.append(np.zeros(1024))
    i = i+1

newDir = os.path.join(os.getcwd(),'masks')
if not os.path.exists(newDir):
    os.mkdir(newDir)

pickle.dump(masker, open( os.path.join(newDir,"mask.p"), "wb" ))
ft_mat = np.array(ft_mat)
labels = np.array(labels)
np.save(os.path.join(newDir,"label.npy"), labels)
np.save(os.path.join(newDir,"ft_mat.npy"), ft_mat)