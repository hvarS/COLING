from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
import os
from engine import Engine
from optimizers import adabound
from utilities import CustomGraphDataset
from models.gat import GAT
from config import args
from models.base_models import NCModel
from optimizers.adabound import AdaBound
from statistics import mean
import optuna

os.environ["CUDA_VISIBLE_DEVICES"]="2"

# Training settings
parser = argparse.ArgumentParser()

args.cuda = not args.no_cuda
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
# adj, features, labels, mask, mask_train,val_mask, test_mask = load_data()
data = CustomGraphDataset()
dataset = data.get_data()
# Model and optimizer
if args.model == 'GAT':
    model = GAT(nfeat=dataset['features'].shape[1], 
                nhid=args.hidden, 
                nclass=2, 
                dropout=args.dropout, 
                nheads=args.n_heads, 
                alpha=args.alpha)
elif args.model == 'HGCN':
    Model = NCModel
    print(args)
    args.n_classes = 2
    args.n_nodes, args.feat_dim = dataset['features'].shape
    # print(hgcn_args)
    model = Model(args)

if args.cuda:
    model.cuda()
    dataset['features'] = dataset['features'].cuda()
    dataset['adj'] = dataset['adj'].cuda()
    dataset['labels'] = dataset['labels'].cuda()


def objective(trial):
    # optimizer = optim.Adam(model.parameters(), 
    #                     lr=args.lr, 
    #                     weight_decay=args.weight_decay)
    args.lr = trial.suggest_loguniform("lr", 1e-8, 1e-1)
    optimizer = AdaBound(model.parameters(),args.lr,gamma=args.gamma)
    loss_values = []
    acc_scores = []
    f1_scores = []
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)
    engine = Engine(args,model,optimizer)
    for epoch in range(1000):
        loss_values.append(engine.train(dataset))
        if epoch%100==0:
            acc_test,f1=engine.compute_test(dataset)
            acc_scores.append(acc_test.item())
            f1_scores.append(f1.item())
            # torch.save(model.state_dict(), 'saved_models/{}.pkl'.format(acc_test))
        scheduler.step()
    return mean(f1_scores)


if __name__=="__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    print("Optimization Finished!")
