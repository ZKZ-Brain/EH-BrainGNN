import torch
from torch.utils.data import  Subset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import numpy as np
from scipy.spatial.distance import pdist, squareform
from torch_geometric.utils import dense_to_sparse
eps = 1e-8
def binarize_top_percent(fc, percent):
    N = fc.shape[0]
    tri_i, tri_j = torch.triu_indices(N, N, 1)      
    vals = fc.abs()[tri_i, tri_j]
    k = int(len(vals) * percent / 100)              
    thresh = torch.topk(vals, k).values.min()        
    mask = fc.abs() >= thresh
    adj = mask.float()
    adj.fill_diagonal_(0)
    return adj

def load_dataset(graph):
    print('loading data')
    num_graphs=graph["label"].size
    print(num_graphs)
    label1=graph["label"]
    label=np.append(label1,label1)
    data_list = []
    for i in range(num_graphs):
        node_features = torch.FloatTensor(graph["graph_struct"][0][i][1])
        #node_features = (node_features - torch.mean(node_features))/(torch.std(node_features)+eps)
        adj = binarize_top_percent(node_features, 20)
        data_example = Data(x=node_features,edge_index=dense_to_sparse(adj)[0],y=label[i])
        data_list.append(data_example)

    return data_list


def BAnotif_load_dataset(graph):
    print('loading data')
    data_list = []
    for i in range(len(graph)):
        g,label = graph[i]
        adj = g.adjacency_matrix(transpose=True)._indices()
        node_features = g.ndata['feat']
        #node_features = (node_features - torch.mean(node_features))/(torch.std(node_features)+eps)
        data_example = Data(x=node_features,edge_index=adj,y=label[0])
        data_list.append(data_example)

    return data_list

def _to_1d(a):
    a = np.asarray(a).reshape(-1)
    return a


def brier_score(p, y):
    p = _to_1d(p).astype(float)
    y = _to_1d(y).astype(float)
    return float(np.mean((p - y) ** 2))

def expected_calibration_error(p, y, n_bins=10, strategy="uniform"):
    """
    ECE for binary classification.
    strategy:
      - 'uniform': equal-width bins in [0,1]
      - 'quantile': equal-count bins (by predicted probability)
    Returns: ece, bin_stats(dict)
    """
    p = _to_1d(p).astype(float)
    y = _to_1d(y).astype(int)

    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
        if len(edges) - 1 < 2:
            # fallback
            edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'.")

    bin_acc = []
    bin_conf = []
    bin_count = []

    ece = 0.0
    n = len(y)

    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            idx = (p >= lo) & (p <= hi)
        else:
            idx = (p >= lo) & (p < hi)

        cnt = int(np.sum(idx))
        bin_count.append(cnt)
        if cnt == 0:
            bin_acc.append(np.nan)
            bin_conf.append(np.nan)
            continue

        conf = float(np.mean(p[idx]))
        acc = float(np.mean(y[idx]))
        bin_conf.append(conf)
        bin_acc.append(acc)

        ece += (cnt / n) * abs(acc - conf)

    stats = {
        "edges": edges,
        "bin_acc": np.array(bin_acc, dtype=float),
        "bin_conf": np.array(bin_conf, dtype=float),
        "bin_count": np.array(bin_count, dtype=int),
    }
    return float(ece)

def pairwise_distances(x):
    #x should be two dimensional
    if x.dim()==1:
        x = x.unsqueeze(1)
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

def calculate_sigma(Z_numpy):   

    if Z_numpy.dim()==1:
        Z_numpy = Z_numpy.unsqueeze(1)
    Z_numpy = Z_numpy.cpu().detach().numpy()
    #print(Z_numpy.shape)
    k = squareform(pdist(Z_numpy, 'euclidean'))       # Calculate Euclidiean distance between all samples.
    sigma = np.mean(np.mean(np.sort(k[:, :10], 1))) 
    if sigma < 0.1:
        sigma = 0.1
    return sigma 

def calculate_gram_mat(x, sigma):
    dist= pairwise_distances(x)
    #dist = dist/torch.max(dist)
    return torch.exp(-dist /sigma)

def reyi_entropy(x,sigma):
    alpha = 1.01
    k = calculate_gram_mat(x,sigma)
    k = k/(torch.trace(k)+eps)
    eigv = torch.abs(torch.linalg.eigh(k)[0])
    eig_pow = eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))
    return entropy


def joint_entropy(x,y,s_x,s_y):
    alpha = 1.01
    x = calculate_gram_mat(x,s_x)
    y = calculate_gram_mat(y,s_y)
    k = torch.mul(x,y)
    k = k/(torch.trace(k)+eps)
    eigv = torch.abs(torch.linalg.eigh(k)[0])
    eig_pow =  eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))

    return entropy

def joint_entropy3(x,y,z,s_x,s_y,s_z):
    alpha = 1.01
    x = calculate_gram_mat(x,s_x)
    y = calculate_gram_mat(y,s_y)
    z = calculate_gram_mat(z,s_z)
    k = torch.mul(x,y)
    k = torch.mul(k,z)
    k = k/(torch.trace(k)+eps)
    eigv = torch.abs(torch.linalg.eigh(k)[0])
    eig_pow =  eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))

    return entropy


def calculate_conditional_MI(x,y,z):

    s_x = calculate_sigma(x)**2
    s_y = calculate_sigma(y)**2
    s_z = calculate_sigma(z)**2
    Hyz = joint_entropy(y,z,s_y,s_z)
    Hxz = joint_entropy(x,z,s_x,s_z)
    Hz = reyi_entropy(z,sigma=s_z)
    Hxyz = joint_entropy3(x,y,z,s_x,s_y,s_z)
    CI = Hyz + Hxz - Hz - Hxyz
    
    return CI

def calculate_MI(x,y):

    s_x = calculate_sigma(x)
    s_y = calculate_sigma(y)
    Hx = reyi_entropy(x,s_x**2)
    Hy = reyi_entropy(y,s_y**2)
    Hxy = joint_entropy(x,y,s_x**2,s_y**2)
    Ixy = Hx + Hy - Hxy
    
    return Ixy

def calculate_single_TC(x):
    num_feature = x.size(1)
    HC = 0.0
    for i in range(num_feature):
        sigma = calculate_sigma(x[:,i])
        HC = HC + reyi_entropy(x[:,i],sigma)
    sigma = calculate_sigma(x)
    Hx= reyi_entropy(x,sigma**2)
    TC = HC - Hx
    return TC

def calculate_Condition_TC(x,y):
    num_feature = x.size(1)
    HC = 0.0
    sigmay = calculate_sigma(y)**2
    for i in range(num_feature):
        sigmax = calculate_sigma(x[:,i])**2
        HC = HC + joint_entropy(x[:,i],y,sigmax,sigmay) - reyi_entropy(y,sigmay)

    sigmax = calculate_sigma(x)**2
    Hxy = joint_entropy(x,y,sigmax,sigmay) - reyi_entropy(y,sigmay)
    TC = HC - Hxy
    return TC

def calculate_TC(x,y):
    TCx = calculate_single_TC(x)
    TCxy = calculate_Condition_TC(x,y)
    return TCx - TCxy

def MI_Est(discriminator, embeddings, positive, batch_size):

    shuffle_embeddings = embeddings[torch.randperm(batch_size)]
    joint = discriminator(embeddings,positive)
    margin = discriminator(shuffle_embeddings,positive)
    mi_est = torch.mean(joint) - torch.log(torch.mean(torch.exp(margin)))

    return mi_est


def get_dataloader_Japan_site(dataset,dataset1, site, site_idx,batch_size):
    
    site = torch.LongTensor(site)
    test_idx = torch.nonzero(site == site_idx)[:,0].numpy().tolist()
    print(site.shape)
    test = Subset(dataset1, test_idx)
    print(test_idx)

    dataloader = dict()
    test_batch_size = len(test_idx)
    print(test_batch_size)
    dataloader['train'] = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader['test'] = DataLoader(test, batch_size=1, shuffle=False)
    return dataloader
