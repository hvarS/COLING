import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utilities import accuracy
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import sys

class Engine:
    def __init__(self,args,model,optimizer) -> None:
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.epoch = 0

    def train(self,dataset):
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        if self.args.model=='GAT':
            output = self.model(dataset['features'], dataset['adj'])
        else:
            # print(dataset['features'].device, dataset['adj'].device)
            output = self.model.decode(self.model.encode(dataset['features'], dataset['adj']),dataset['adj'])
        # print('Forward Succesful')
        # print(output.shape)
        # print(output[0:5])
        # sys.exit(0)
        tweet_mask = dataset['tweet_mask']
        train_mask = dataset['train_mask']
        val_mask = dataset['val_mask']
        labels = dataset['labels']
        loss_train = F.nll_loss(output[tweet_mask][train_mask], labels[train_mask])
        acc_train = accuracy(output[tweet_mask][train_mask], labels[train_mask])
        loss_train.backward()
        self.optimizer.step()

        if not self.args.fastmode:
            self.model.eval()
            if self.args.model =='GAT':
                output = self.model(dataset['features'], dataset['adj'])
            else:
                output = self.model.decode(self.model.encode(dataset['features'], dataset['adj']),dataset['adj'])
        loss_val = F.nll_loss(output[tweet_mask][val_mask], labels[val_mask])
        acc_val = accuracy(output[tweet_mask][val_mask], labels[val_mask])
        print('Epoch: {:04d}'.format(self.epoch+1),
            'loss_train: {}'.format(loss_train.data.item()),
            'acc_train: {}'.format(acc_train.data.item()),
            'loss_val: {}'.format(loss_val.data.item()),
            'acc_val: {}'.format(acc_val.data.item()),
            'time: {:.4f}s'.format(time.time() - t))
        self.epoch+=1
        return loss_val.data.item()


    def compute_test(self,dataset):
        self.model.eval()
        if self.args.model=='GAT':
            output = self.model(dataset['features'],dataset['adj'])
        else:
            output = self.model.decode(self.model.encode(dataset['features'], dataset['adj']),dataset['adj'])
        tweet_mask = dataset['tweet_mask']
        test_mask = dataset['test_mask']
        labels = dataset['labels']
        loss_test = F.nll_loss(output[tweet_mask][test_mask], labels[test_mask])
        print(classification_report(labels[test_mask].cpu().numpy(),np.argmax(output[tweet_mask][test_mask].cpu().detach().numpy(),-1)))
        print(confusion_matrix(labels[test_mask].cpu().numpy(),np.argmax(output[tweet_mask][test_mask].cpu().detach().numpy(),-1)))
        acc_test = accuracy(output[tweet_mask][test_mask], labels[test_mask])
        print(accuracy_score(labels[test_mask].cpu().numpy(),np.argmax(output[tweet_mask][test_mask].cpu().detach().numpy(),-1)))
        print("Test set results:",
            "loss= {:.4f}".format(loss_test),
            "accuracy= {:.4f}".format(acc_test))
        # np.save('saved_models/{}.npy'.format(acc_test), confusion_matrix(labels[test_mask].cpu().numpy(),np.argmax(output[tweet_mask][test_mask].cpu().detach().numpy(),-1)))          
        return acc_test