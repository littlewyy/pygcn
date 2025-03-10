import numpy as np
import scipy.sparse as sp
import torch

#为labels生成对应的独热码向量
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
    #print(idx_features_labels)
    #print(idx_features_labels[:,1:-1])
    # 调试信息：idx_features_labels[:,1:-1]，读取矩阵每一行中，第1维到第-1维的内容，其中最后一列为第-1维。1：-1为左闭右开区间，不包括第-1维
    # 数据cora.content中，第0维为论文编号，第1维到第-1维之间为论文词向量，第-1维（也就是最后一维）为论文所属领域（标签）
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32) #转换成行压缩稀疏矩阵
    #print(features)
    labels = encode_onehot(idx_features_labels[:, -1]) #将论文标签转化为独热码向量

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32) #获取论文编号
    idx_map = {j: i for i, j in enumerate(idx)} #论文编号：序号（从0开始依次），序号相当于为后续论文编号离散化做准备
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32) #读取引用关系
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape) #用新序号重新命名边
    
    #print(edges.shape)
    #print(labels.shape)

    #adj：邻接矩阵
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), #shape为点数 * 点数，点数即为论文数目，即为labels总数（含重复）
                        dtype=np.float32) #将边集转化为coo格式的稀疏矩阵。矩阵中元素数据值为1。

    #将邻接矩阵变换为对称邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features) #特征归一化：不理解！！！！！
    adj = normalize(adj + sp.eye(adj.shape[0])) #邻接矩阵归一化：A' = A + I   A' = D'^-1 A'

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    #将稀疏矩阵转换为张量
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1]) #什么东西？独热码转成常规label?
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """归一化，D^(-1)点乘A"""
    rowsum = np.array(mx.sum(1)) #计算度
    r_inv = np.power(rowsum, -1).flatten() #度取倒数
    r_inv[np.isinf(r_inv)] = 0. #将无穷大置为0.（double类型的0）
    r_mat_inv = sp.diags(r_inv) #生成度矩阵
    mx = r_mat_inv.dot(mx) #邻接矩阵点乘度矩阵
    return mx

#计算正确率
def accuracy(output, labels):
    #不确定：output.max(1)，则按照行求最大值，并返回最大值和最大值的索引，此处用.type_as(labels)使得索引类型与labels相同。output.max(0)则按照列求最大值，其余相同。
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double() #返回preds和labels相同的那些值
    correct = correct.sum() #计算相同的数目
    return correct / len(labels) #len(labels)，计算labels中元素个数，依次计算准确率

#把稀疏矩阵转换成张量函数。张量函数由坐标indices,数值values和尺寸shape组成。
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32) #.tocoo()转化成coo格式的张量，.astype()指定数据类型
    #torch.from_numpy()：从numpy类型的数据中获取数据
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)) #np.vstack((a,b))，将a,b中的元素按顺序排列，成为一个新的整体。.astype()指定数据类型。
    #此处，将sparse_mx每一维的坐标信息依次排列，作为indices
    values = torch.from_numpy(sparse_mx.data) #将数据从sparse_mx中取出，作为张量的数据
    shape = torch.Size(sparse_mx.shape) #torch.Size()，计算尺寸大小
    return torch.sparse.FloatTensor(indices, values, shape)
