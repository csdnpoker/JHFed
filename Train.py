import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from Split import Split


KL_Loss = nn.KLDivLoss(reduction='batchmean')
Softmax = nn.Softmax(dim=1)
LogSoftmax = nn.LogSoftmax(dim=1)
CE_Loss = nn.CrossEntropyLoss()


def train_normal(node):
    node.model.to(node.device).train()
    train_loader = node.train_data
    total_loss = 0.0
    avg_loss = 0.0
    correct = 0.0
    acc = 0.0
    description = "Training (the {:d}-batch): tra_Loss = {:.4f} tra_Accuracy = {:.2f}%"
    with tqdm(train_loader) as epochs:
        for idx, (data, target) in enumerate(epochs):
            node.optimizer.zero_grad()
            epochs.set_description(description.format(idx + 1, avg_loss, acc))
            data, target = data.to(node.device), target.to(node.device)
            output = node.model(data)
            loss = CE_Loss(output, target)
            loss.backward()
            node.optimizer.step()
            total_loss += loss
            avg_loss = total_loss / (idx + 1)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum()
            acc = correct / len(train_loader.dataset) * 100


def train_avg(node):
    node.model.to(node.device).train()
    train_loader = node.train_data
    total_loss = 0.0
    avg_loss = 0.0
    correct = 0.0
    acc = 0.0
    description = "Node{:d}: loss={:.4f} acc={:.2f}%"
    with tqdm(train_loader) as epochs:
        for idx, (data, target) in enumerate(epochs):
            node.optimizer.zero_grad()
            epochs.set_description(description.format(node.num, avg_loss, acc))
            data, target = data.to(node.device), target.to(node.device)
            output = node.model(data)
            loss = CE_Loss(output, target)
            loss.backward()
            node.optimizer.step()
            total_loss += loss
            avg_loss = total_loss / (idx + 1)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum()
            acc = correct / len(train_loader.dataset) * 100



class Trainer(object):

    def __init__(self, args):
        if args.algorithm == 'normal':
            self.train = train_normal
        elif args.algorithm == 'fed_avg':
            self.train = train_avg

    def __call__(self, node):
        self.train(node)



def loss_coteaching(y_1, y_2, t, forget_rate, args):
    if args.loss == 'CE':
        loss_1 = F.cross_entropy(y_1, t, reduce=False)
        loss_2 = F.cross_entropy(y_2, t, reduce=False)

    elif args.loss == 'GCE':
        loss_1 = GCELoss(y_1, t)
        loss_2 = GCELoss(y_2, t)

    loss_1 = F.cross_entropy(y_1, t, reduce=False)
    ind_1_sorted = torch.argsort(loss_1.data).cuda()


    ind_2_sorted = torch.argsort(loss_2.data).cuda()

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * ind_1_sorted.size()[0])

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]

    a = ind_1_update.tolist()
    b = ind_2_update.tolist()
    ovellap = len(set(a) & set(b))

    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])
    return torch.sum(loss_1_update), torch.sum(loss_2_update), ovellap





