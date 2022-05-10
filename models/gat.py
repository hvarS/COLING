import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.graphAttentionLayer import GraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        # print('dropout done')
        # print(x.device,adj.device)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # print('cat done')
        # print(x.shape)
        x = F.dropout(x, self.dropout, training=self.training)
        # print('dropout-2 done')
        x = F.elu(self.out_att(x, adj))
        # print('elu done')
        return F.log_softmax(x, dim=1)


# class HGCN(nn.Module):
#     def __init__(self, args, nfeat, nhid, nclass, dropout, alpha, nheads):
#         """Dense version of Hyperbolic Convolution Layer"""
#         super(HGCN, self).__init__()
#         #manifold can be Euclidean ,PoincareBall
        
#         self.manifold = getattr(manifolds, args)()
#         self.dropout = dropout
#         dims, acts, self.curvatures = get_dim_act_curv(args)
#         c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
#         self.convs = [HyperbolicGraphConvolution(self.manifold,nfeat, nhid, c_in, c_out, args.dropout, act, args.bias, args.use_att, args.local_agg) for _ in range(nheads)]
#         for i, conv in enumerate(self.convs):
#             self.add_module('hconv_{}'.format(i), conv)

#         self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)
#         # print('dropout done')
#         # print(x.device,adj.device)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         # print('cat done')
#         # print(x.shape)
#         x = F.dropout(x, self.dropout, training=self.training)
#         # print('dropout-2 done')
#         x = F.elu(self.out_att(x, adj))
#         # print('elu done')
#         return F.log_softmax(x, dim=1)



# g = GAT(768, 8, 2, 0.2, 5, 8)
