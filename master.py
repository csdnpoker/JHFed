import torch
import pickle
import socket
from pyspark improt SparkContext
from datatime import Datatime
from hdfs import Client
from Split import Split, Global_Node
from Args import args_parser
from Data import Data
from Train import Trainer
import copy

address = ("192.168.43.185",10000)#本主机IP
readdr1 = ("192.168.43.83",10000)#客户端主机IP
readdr2 = ("192.168.43.68",10000)#客户端主机IP
readdr3 = ("192.168.43.138",10000)#客户端主机IP
readdr4 = ("192.168.43.232",10000)#客户端主机IP
readdr5 = ("192.168.43.150",10000)#客户端主机IP
readdr6 = ("192.168.43.48",10000)#客户端主机IP
readdr7 = ("192.168.43.164",10000)#客户端主机IP
readdr8 = ("192.168.43.208",10000)#客户端主机IP
s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
s.bind(address)
cohdfs = Client('http://192.168.43.185:50070',root='/user/firefly',timeout=1000, session=False)

if __name__ == '__main__':

    args = args_parser()
    Train = Trainer(args)
    SFL_num=args.sflnum
    Data = Data(args)
    Global_node = Global_Node(Data.test_all, args)
    Edge_nodes = [Split(k, Data.train_loader[k], Data.test_loader, args) for k in range(args.node_num)]
    optim=args.optim
    for rounds in range(args.R):
        Datatime.work(Edge_nodes,Global_node)
        Global_node.processing()
        Global_list = (copy.deepcopy(Global_node.model.state_dict()))
        while True:
            data = input("input:")
            if data == "ready":
                with cohdfs.write('/user/firefly/master/globalmodel.pth') as f:
                    pickle.dump(Global_list, f)
                break
        match=Datatime.optimal(optim)
        Datatime.matchresult(match)
s.close()
