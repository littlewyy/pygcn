import numpy as np
import scipy.sparse as sp
import torch

def encode_onehot(labels):
    '''
    目的：为labels创建独热码向量（labels为可重复的字符串，应给每种label分配一个整数，以便映射到对应的独热码向量）
    思路：首先去重；然后分配整数；最后转化成独热码向量
    
    函数用法：
    set(seq = ())：创建不重复的集合（如果seq为空，则创建空集；否则将seq中元素去重后存入）
    np.identity(n)：创建 n * n 的单位矩阵
    dict = {idx:val for ... in ...  if ... in ...}
    enumerate(iterables,start = 0): 创建从start开始与iterables逐个对应的映射函数（start默认值为0）
    np.array(p_object,dtype = None)：创建一个数组，其内容为p_object，类型为dtype
    map(func,iterables)：创建映射函数
    list():将元素整合成序列
    '''
    
    classes = set(labels) #labels去重
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)} #给去重后的labels对应分配独热码向量。enumerate起到分配整数的作用，独热码向量视为单位矩阵中的一行。 
                    
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    #在字典中，取出labels对应独热码向量（通过map函数实现，map(a,b)表示从函数a中查找索引为b的键值），然后转换成序列，再用np.array转换成数组，用dtype将数组元素置为int
                              
    return labels_onehot

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
