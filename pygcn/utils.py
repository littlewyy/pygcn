import numpy as np
import scipy.sparse as sp
import torch

def encode_onehot(labels):
    '''
    目的：为labels创建独热码向量
    思路：
    先给每个labels分配一个元素，
    然后根据这些元素的顺序，对应地给元素分配独热码向量，
    最后再把原label跟独热码向量做映射
    
    函数用法：
    set(seq = ())：为seq (元素序列）对应创建互不相同的元素
    np.identity(n)：创建 n * n 的单位矩阵
    dict = {idx:val for ... in ...  if ... in ...}
    enumerate(iterables,start = 0): 创建从start开始与iterables逐个对应的映射函数。
    np.array(p_object,dtype = None)：创建一个数组，其内容为p_object，类型为dtype
    map(func,iterables)：创建映射函数
    list():将元素整合成序列
    '''
    
    classes = set(labels) #给每个label都分配一个独特的元素，存在classes里面
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)} 
                    
    '''
    创建一个字典，其中索引值为别名，键值为对应独热码向量
      独热码向量从单位向量中（通过映射函数左值）获取对应行
    '''
    
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    #先把独热码向量和元素做映射，然后转换成序列，再用np.array转换成数组，用dtype将数组元素置为int
                              
    return labels_onehot
'''
    疑问：一开始为什么要用set生成对应的互不相同元素？是为了去重吗？
    直接用原labels对应独热码向量不行吗，何必多此一举？
   
'''


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
