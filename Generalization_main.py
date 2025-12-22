import numpy as np
import scipy.io as sio
import os
import os.path as osp
import torch
import torch.nn as nn
import argparse
from scipy.io import loadmat
from utilis import load_dataset,get_dataloader_Japan_site
import random
from sklearn.metrics import confusion_matrix, roc_auc_score
criterion = nn.CrossEntropyLoss()



parser = argparse.ArgumentParser(description='Graph Generative causal explanations')
parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
parser.add_argument("--latent_dim", type=int, default=  [128,128,128,128,128], help="classifier hidden dims")
parser.add_argument('--readout', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over nodes in a graph: sum or average')
parser.add_argument('--dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
parser.add_argument("--mlp_hidden", type=int, default=  [64,64], help="mlp hidden dims")
parser.add_argument("--emb_normlize", type = bool, default=  False, help="mlp hidden dims")
parser.add_argument("--adj_normlize", type = bool, default=  True, help="mlp hidden dims")
parser.add_argument('--epochs', type=int, default=350,
                        help='number of epochs to train (default: 100)')
parser.add_argument("--weight_decay", type=float, default=0.0005, help="Adam weight decay. Default is 5*10^-5.")
args = parser.parse_args()

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#device = 'cpu'
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# --- load data ---
from utils import get_dataloader,expected_calibration_error
import torch

graph_filename = "F:\GNN_MDD\data\China_MDD.mat"
graph = loadmat(graph_filename)
dataset = load_dataset(graph)

graph_filename1 = "data\Japan_MDD.mat"
graph1 = loadmat(graph_filename1)
dataset1 = load_dataset(graph1)

site = torch.zeros(446).long()
site[0:125]=1
site[125:190]=2
site[190:252]=3
site[252:446]=4

# --- load classifier ---
from GIN_classifier import GINNet
input_dim = 116
output_dim = 2
print(input_dim)
print(output_dim)
classifier = GINNet(input_dim, output_dim, args, device).to(device)
Softmax= nn.Softmax(dim=-1)


# --- train/load Explainer model---
from SubgraphGenerator import SubgraphGenerator
Exp = SubgraphGenerator(classifier, device).to(device)

def test(data, Exp, device): 
    classifier.eval()
    Exp.eval()
    acc_accum = 0
    auc_accum = 0
    sen_accum = 0
    spc_accum = 0
    prc_accum = 0
    f1s_accum = 0
    mcc_accum = 0
    num = 0
    for da in data: 

        logits, loss = Exp(da)
        labels = torch.LongTensor(da.y).to(device)
        pred = logits.max(1, keepdim=True)[1]
        correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
        pred = pred.cpu().numpy()
        labels = labels.cpu().numpy()
        if len(labels)>1:
            test_acc, test_auc , test_sen, test_spc, test_prc, test_f1s, test_mcc = calc_performance_statistics(pred,labels)
            probs = Softmax(logits)
            ses = expected_calibration_error(probs[:,0].cpu().detach().numpy(),labels)
            acc = correct / float(len(da.y))
            acc_accum = acc_accum + acc
            auc_accum = auc_accum + test_auc
            sen_accum = sen_accum + test_sen
            spc_accum = spc_accum + test_spc
            prc_accum = prc_accum + test_prc
            f1s_accum = f1s_accum + test_f1s
            mcc_accum = mcc_accum + test_mcc
            num = num + 1
    acc_test = acc_accum/num
    auc_test = auc_accum/num
    sen_test = sen_accum/num
    spc_test = spc_accum/num
    prc_test = prc_accum/num
    f1s_test = f1s_accum/num
    mcc_test = mcc_accum/num

    return acc_test, auc_test, sen_test, spc_test, prc_test, f1s_test, mcc_test,ses

def calc_performance_statistics(y_pred, y):

    TN, FP, FN, TP = confusion_matrix(y, y_pred).ravel()
    N = TN + TP + FN + FP
    S = (TP + FN) / N
    P = (TP + FP) / N
    acc = (TN + TP) / N
    sen = TP / (TP + FN)
    spc = TN / (TN + FP)
    prc = TP / (TP + FP)
    f1s = 2 * (prc * sen) / (prc + sen)
    mcc = (TP / N - S * P) / np.sqrt(P * S * (1 - S) * (1 - P))
    auc = roc_auc_score(y,y_pred)

    return acc, auc, sen, spc, prc, f1s, mcc

for site_idx in range(0,4):
    dataloader = get_dataloader_Japan_site(dataset,dataset1, site, site_idx, args.batch_size)
    opt_params = list(Exp.parameters()) + list(classifier.parameters())
    opt = torch.optim.Adam(opt_params, lr=args.lr , weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)
    for k in range(0, args.epochs):
        classifier.train()
        Exp.train()
        acc_accum = 0
        num = 0
        for data in dataloader['train']:

            logits, loss = Exp(data)

            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()

            labels = torch.LongTensor(data.y).to(device)

            pred = logits.max(1, keepdim=True)[1]
            correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
            acc = correct / float(len(data.y))
            acc_accum = acc_accum + acc
            num = num + 1

        acc_train = acc_accum/num
        scheduler.step()
        print("accuracy train: %f" %(acc_train))

        acc_test, auc_test, sen_test, spc_test, prc_test, f1s_test, mcc_test,ses = test(dataloader['test'], Exp, device)
        print("accuracy test: %f" %(acc_test))

        filename="MDD_Japan_site_layer" +  str(site_idx) +  ".txt"
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write("%f %f %f %f %f %f %f %f %f %f" % (loss, acc_train, acc_test, auc_test, sen_test, spc_test, prc_test, f1s_test, mcc_test,ses))
                f.write("\n")
        else:
            with open(filename, 'a+') as f:
                f.write("%f %f %f %f %f %f %f %f %f %f" % (loss, acc_train, acc_test, auc_test, sen_test, spc_test, prc_test, f1s_test, mcc_test,ses))
                f.write("\n")

    classifier = GINNet(input_dim, output_dim, args, device).to(device)
    Exp = SubgraphGenerator(classifier, device).to(device)



