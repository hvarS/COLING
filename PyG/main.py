from dataset import data
from models.gcn import GCN
import torch 
import torch.nn.functional as F
from visual import visualize
import sys

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

model = GCN(data.num_node_features,512,2)
model.to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.eval()
out = model(data.x, data.edge_index)
visualize(out.cpu(), data.y.cpu(), 'before_training.png')


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
      return test_acc


for epoch in range(1, 101):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
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

test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')

model.eval()
out = model(data.x, data.edge_index)
visualize(out.cpu(), data.y.cpu(),'after_training.png')