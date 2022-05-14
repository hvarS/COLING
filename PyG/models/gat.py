import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class GAT(torch.nn.Module):
    def __init__(self, in_dim, hidden_channels, num_classes,heads = 8):
        super().__init__()
        torch.manual_seed(1234567)

        self.conv1 = GATv2Conv(in_dim, hidden_channels,heads )
        self.conv2 = GATv2Conv(hidden_channels*heads,num_classes, heads)
        self.linear = torch.nn.Linear(num_classes*heads,num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x,p = 0.5, training = self.training)
        x = self.linear(x)
        return F.log_softmax(x,dim=1)