from __future__ import division #导入精确除法（否则/只能为整除）
from __future__ import print_function #导入print()，即print必须加括号

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

#如果有gpu,用gpu
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

#指定生成随机数的种子，从而每次生成的随机数都是相同的。
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda: #如果有cuda,应用cuda生成随机数
    torch.cuda.manual_seed(args.seed)

'''
开始训练
'''

# 载入数据
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
# 模型
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1, #max().item()什么意思
            dropout=args.dropout)
# 优化器
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

#如果有cuda，则用cuda
if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    #将模型转为训练模式
    model.train()
    #将梯度置零。（每一轮batch开始时需要设置）
    optimizer.zero_grad()
    #应用模型
    output = model(features, adj)
    #训练集损失函数
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    #准确率
    acc_train = accuracy(output[idx_train], labels[idx_train])
    #根据损失函数反向求导
    loss_train.backward()
    #更新所有参数，实现优化
    optimizer.step()

    if not args.fastmode: #args.fastmode什么意思？
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval() #将模型转变为验证模式
        output = model(features, adj)

    #验证集的损失函数
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

#定义测试函数
def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
# 逐个epoch进行train
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# 最后test
test()
