import copy
import torch
import random
import Model

def init_model(model_type):
    model = []
    if model_type == 'LeNet5':
        model = Model.LeNet5()
    elif model_type == 'ResNet18':
        model = Model.ResNet18()
    elif model_type == 'CNN':
        model = Model.CNN()
    return model


def init_optimizer(model, args):
    optimizer = []
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    return optimizer


def weights_zero(model):
    for p in model.parameters():
        if p.data is not None:
            p.data.detach_()
            p.data.zero_()


class Split(object):
    def __init__(self, num, train_data, test_data, args):
        self.args = args
        self.num = num + 1
        self.device = self.args.device
        self.train_data = train_data
        self.test_data = test_data

        self.model = init_model(self.args.local_model).to(self.device)
        self.optimizer = init_optimizer(self.model, self.args)


    def fork(self, global_node):
        self.model = copy.deepcopy(global_node.model).to(self.device)
        self.optimizer = init_optimizer(self.model, self.args)



class Global_Node(object):
    def __init__(self, test_data, args):
        self.num = 0
        self.args = args
        self.device = self.args.device
        self.model = init_model(self.args.global_model).to(self.device)

        self.test_data = test_data
        self.Dict = self.model.state_dict()

#         self.edge_node = [Model.ResNet18().to(self.device) for k in range(args.node_num)]
        self.edge_node = [init_model(self.args.global_model).to(self.device) for k in range(args.node_num)]
        self.init = False
        self.save = []

    def merge(self, Edge_nodes):
        Node_State_List = [copy.deepcopy(Edge_nodes[i].model.state_dict()) for i in range(len(Edge_nodes))]
        torch.save(Node_State_List[0],'./test22.txt')
        self.Dict = torch.load('./test22.txt')

        for key in self.Dict.keys():
            for i in range(1, len(Edge_nodes)):
                self.Dict[key] += Node_State_List[i][key]

            self.Dict[key] = self.Dict[key].float()
            self.Dict[key] /= len(Edge_nodes)
        self.model.load_state_dict(self.Dict)


    def update(self, Edge_node):
        self.edge_node[Edge_node.num-1] = Edge_node.model

    def init_processing(self):
        assert self.init
        ## warmup
        Node_State_List = [copy.deepcopy(self.edge_node[i].state_dict()) for i in self.save]
        self.Dict = Node_State_List[0]
        for key in self.Dict.keys():
            if 'num_batches_tracked' in key:
                continue

            for i in range(1, len(Node_State_List)):
                self.Dict[key] += Node_State_List[i][key]

            self.Dict[key] /= float(len(Node_State_List))

        self.model.load_state_dict(self.Dict)

    def DFL(self,DFLmodel):
        self.model.load_state_dict(DFLmodel)
    def processing(self):
        Node_State_List = [copy.deepcopy(self.edge_node[i].state_dict()) for i in range(self.args.node_num)]
        self.Dict = Node_State_List[0]
        for key in self.Dict.keys():
            if 'num_batches_tracked' in key:
                continue
            for i in range(1, self.args.node_num):
                self.Dict[key] += Node_State_List[i][key]
            self.Dict[key] /= self.args.node_num
        self.model.load_state_dict(self.Dict)

