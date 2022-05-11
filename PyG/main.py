from dataset import data
from models.gcn import GCN
import torch 
import torch.nn.functional as F

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

model = GCN(in_dim = data.num_node_features,num_classes=2).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(1000):
    ## Train Step 
    optimizer.zero_grad()
    out = model(data)
    # print(out.shape)
    loss = F.nll_loss(out[data.trn_mask], data.y[data.trn_mask])
    loss.backward()
    optimizer.step()

    ## Val Step 
    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(dim=1)
        correct = (pred[data.tst_mask] == data.y[data.tst_mask]).sum()
        acc = int(correct) / int(data.tst_mask.sum())
    print(f'Validation Accuracy: {acc:.4f}')