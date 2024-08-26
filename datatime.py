import time
import socket
import torch
import numpy as np
from SA import SA, func
import random
cohdfs = Client('http://192.168.43.185:50070',root='/user/firefly',timeout=1000, session=False)
import pickle
file='/user/firefly/slave'
name='/model.pth'
from Node import Split, Global_Node
address = ('192.168.43.185',10000)#本主机IP
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
ipl=["192.168.43.83","192.168.43.68","192.168.43.138","192.168.43.232","192.168.43.150","192.168.43.48","192.168.43.164","192.168.43.208"]
ip=[("192.168.43.83",10000),("192.168.43.68",10000),("192.168.43.138",10000),("192.168.43.232",10000),("192.168.43.150",10000),("192.168.43.48",10000),("192.168.43.164",10000),("192.168.43.208",10000)]
def selectRandom(value):
    return random.choice(value)
class Datatime(object):
    def __init__(self, args):
        self.args=args
        self.datalist=[0 for index in range(8)]
        self.listnum=0
        self.timelist= [0 for index in range(8)]
        self.slist=[]
        self.plist=[]
        self.nodenum=args.nodenum
        self.b1=args.b1
        self.b2=args.b2
    def work(self,Edge_nodes,Global_node):
        while True:
            data, addr = s.recvfrom(2048)
            time_start = time.time()
            if data.startswith('slave'):
                list = int(data[5])-1
                self.datalist[list] +=1
                self.listnum +=1
                with cohdfs.read(f"{file}{int(data[5])}{name}") as f:
                    Edge_nodes[list] = torch.load(pickle.loads(f.read()))
                Global_node.update(Edge_nodes[list])
            time_end = time.time()
            time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
            time_now = time_sum
            self.timelist[list] += time_sum
            if (time_now + time_sum/self.datalist[list]) > self.b1:
                self.slist.append(list)
            if (time_now + time_sum/self.datalist[list]) < self.b2:
                self.plist.append(list)
            if self.listnum == self.nodenum:
                break


    def optimal(self, optim):
        lista=[]
        if optim == 'SA':
            sa = SA(func)
            sa.run()
        elif optim == 'Random':
            for i in range(len(self.slist)):
                aa = [self.slist[i], selectRandom(self.plist)]
                lista.append(aa)
            return lista
        elif optim =='None':
            return None

    def matchresult(self, match):
        if match != 'None':
            for i in range (len(match)):
                st=self.slist.index([i][0])
                stdata='JHS'
                stip=ipl[st]
                pc=self.slist.index([i][1])
                pcdata = 'JHP'
                pcip = ipl[pc]
                s.sendto(stdata.encode("utf-8"), ip[st])
                s.sendto(pcdata.encode("utf-8"), ip[pc])
                s.sendto(stip.encode("utf-8"), ip[st])
                s.sendto(pcip.encode("utf-8"), ip[pc])


