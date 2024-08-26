import torch
import pickle
import socket
from pyspark improt SparkContext
from hdfs import Client
from Split import Split, Global_Node
from Args import args_parser
from Data import Data
from utils import LR_scheduler, Summary
from Train import Train
import copy
ipl=["192.168.43.83","192.168.43.68","192.168.43.138","192.168.43.232","192.168.43.150","192.168.43.48","192.168.43.164","192.168.43.208"]
cohdfs = Client('http://192.168.43.185:50070',root='/user/firefly',timeout=1000, session=False)
addr=('192.168.43.185',10000)
readdr=('192.168.43.68',10000)
file='/user/firefly/slave'
name='/model.pth'
ip=''
dflip=''
newreaddr=(f"{ip}",10000)
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
s.bind(readdr)
if __name__ == '__main__':
    connection = 'CFL'
    args = args_parser()
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.split = args.node_num
    args.global_model = args.local_model
    print('Running on', args.device)
    Data = Data(args)
    Train = Trainer(args)
    Summary(args)
    Edge_nodes = [Split(k, Data.train_loader[k], Data.test_loader, args) for k in range(args.split)]
    Globalmodel = Edge_nodes[0]
    for rounds in range(args.R):
        print('The {:d}-th round'.format(0 + 1))
        LR_scheduler(rounds, Edge_nodes, args)
        Edge_nodes[0].fork(Globalmodel)
        for epoch in range(args.E):
            Train(Edge_nodes[0])
        Edge_list=(copy.deepcopy(Edge_nodes[0].model.state_dict()))
        if connection == 'DFL':
            with cohdfs.read(f"{file}{ipl.index(dflip)+1}{name}") as f:
                DFLp = pickle.loads(f.read())
                DFLmodel=torch.load(DFLp)
                DFL_State_List = [copy.deepcopy(DFLmodel.model.state_dict())]
                # self.Dict = Node_State_List[0]

                for key in Edge_list.keys():
                    Edge_list[key] += DFL_State_List[key]

                    Edge_list[key] = Edge_list[key].float()  # 不知道为什么数据类型会发生变化
                    Edge_list[key] /= 2
        while True:
            data = input("input:")
            if data == "ready":
                senddata=f"{'slave5'}"
                s.sendto(senddata.encode("utf-8"), addr)
                with cohdfs.write('/user/firefly/slave05/model.pth') as f:
                    pickle.dump(Edge_list,f)
                break
        while True:
            recivedata, addrg = s.recvfrom(2048)
            if recivedata == 'JHP':
                while True:
                    ip, addrg = s.recvfrom(2048)
                    if ip:
                        dflip=ip

                        break
                with cohdfs.read('/user/firefly/master/globalmodel.pth') as f:
                    Globalmodel = torch.load(pickle.loads(f.read()))
                connection = 'DFL'
                break
            elif recivedata == 'JHS':
                while True:
                    ip, addrg = s.recvfrom(2048)
                    if ip:
                        readdr=newreaddr
                        break
                with cohdfs.read('/user/firefly/slave05/globalmodel.pth') as f:
                    Globalmodel = torch.load(pickle.loads(f.read()))
            elif recivedata == 'continue':
                with cohdfs.read('/user/firefly/slave05/globalmodel.pth') as f:
                    Globalmodel = torch.load(pickle.loads(f.read()))
                connection = 'CFL'
                break
    Summary(args)

