import torch
import Node
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()




def LR_scheduler(rounds, Edge_nodes, args):

    for i in range(len(Edge_nodes)):
        Edge_nodes[i].args.lr = args.lr
        Edge_nodes[i].args.alpha = args.alpha
        Edge_nodes[i].args.beta = args.beta
        Edge_nodes[i].optimizer.param_groups[0]['lr'] = args.lr
    
    print('Learning rate={:.4f}'.format(args.lr))


def Summary(args):
    print("Summary：\n")
    print("max_lost:{}\n".format(args.max_lost))
    print("dataset:{}\tbatchsize:{}\n".format(args.dataset, args.batchsize))
    print("node_num:{},\tsplit:{}\n".format(args.node_num, args.split))
    # print("iid:{},\tequal:{},\n".format(args.iid == 1, args.unequal == 0))
    print("global epochs:{},\tlocal epochs:{},\n".format(args.R, args.E))
    print("global_model:{}，\tlocal model:{},\n".format(args.global_model, args.local_model))
