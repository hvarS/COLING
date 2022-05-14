from dataset import data
from models.gcn import GCN
from models.gat import GAT
import torch 
import torch.nn.functional as F
from visual import visualize
import sys
from sklearn.metrics import f1_score
from tqdm import tqdm 
from optimizers.adabound import AdaBound

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = GAT(data.num_node_features,512,2)
model.to(device)
data = data.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
optimizer = AdaBound(model.parameters(),lr=0.01,weight_decay=5e-4)

# model.eval()
# out = model(data.x, data.edge_index)
# visualize(out.cpu(), data.y.cpu(), 'before_training.png')


def train():
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data.x, data.edge_index)  # Perform a single forward pass.
    #   print(out.shape)
      loss = F.nll_loss(out[data.trn_mask], data.y[data.trn_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss

def test():
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[data.tst_mask] == data.y[data.tst_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(data.tst_mask.sum())  # Derive ratio of correct predictions.
      f1 = f1_score(data.y[data.tst_mask].cpu().numpy(),pred[data.tst_mask].cpu().numpy())
      return test_acc,f1


loop_object = tqdm(range(1,1001))
best = -1
for epoch in loop_object:
    loss = train()
    s = f'Epoch: {epoch:03d}, Loss: {loss:.4f}'
    loop_object.set_description(s)
    if epoch%100==0:
      test_acc,f1 = test()
      best = max(best,f1)
      s = f'Test Accuracy: {test_acc:.4f} , F1 Score: {f1:.4f}'
      loop_object.set_description(s)
# for epoch in range(1000):
#     ## Train Step 
#     model.train()
#     optimizer.zero_grad()
#     out = model(data.x,data.edge_index)
#     # print(out.shape)
#     loss = F.nll_loss(out[data.trn_mask], data.y[data.trn_mask])
#     loss.backward()
#     optimizer.step()

#     ## Val Step 
#     model.eval()
#     with torch.no_grad():
#         pred = model(data.x,data.edge_index).argmax(dim=1)
#         correct = (pred[data.tst_mask] == data.y[data.tst_mask]).sum()
#         acc = int(correct) / int(data.tst_mask.sum())
#     print(f'Validation Accuracy: {acc:.4f}')

print(f'Best Score : {best:.4f}')

# model.eval()
# out = model(data.x, data.edge_index)
# visualize(out.cpu(), data.y.cpu(),'after_training.png')