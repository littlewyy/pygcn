import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 由于weight是可以训练的，因此使用parameter定义
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # bias：偏移量
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features)) #不理解：bias的参数是out_features?
        else: #register_parameter(name,None/para)，给Module添加参数
            self.register_parameter('bias', None)
        #参数初始化
        self.reset_parameters()

    #为了让每次训练产生的初始参数尽可能的相同，从而便于实验结果的复现，可以设置固定的随机数生成种子。
    def reset_parameters(self):
        #.size(1)计算行的数目
        stdv = 1. / math.sqrt(self.weight.size(1))
        #.data.uniform_(a,b)生成(a,b)的随机数
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    #前向传播：A * X * W
    def forward(self, input, adj):
        support = torch.mm(input, self.weight) #torch.mm：矩阵乘法，此处先计算X * W。（区分：torch.mul为矩阵对应位相乘）
        output = torch.spmm(adj, support) #torch.spmm：稀疏矩阵相乘。再计算A * (X * W)
        #在bias不为None时才可以加偏移量
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
