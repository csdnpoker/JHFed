import argparse


def args_parser():
    lister = argparse.ArgumentParser()

    # Total
    lister.add_argument('--algorithm', type=str, default='fed_avg')
    lister.add_argument('--policy', type=str, default='AFL')
    lister.add_argument('--ednode', type=int, default=8)
    lister.add_argument('--split', type=str, default=1)
    lister.add_argument('--device', type=str, default='cuda:0')
    lister.add_argument('--node_num', type=int, default=1)
    lister.add_argument('--R', type=int, default=2)
    lister.add_argument('--optim', type=str, default='SA')
    lister.add_argument('--E', type=int, default=1)
    lister.add_argument('--b1', type=int, default=500)
    lister.add_argument('--b2', type=int, default=200)
    lister.add_argument('--sflnum', type=int, default=8)
    lister.add_argument('--max_lost', type=int, default=1)
    lister.add_argument('--warmup', type=int, default=5)
    lister.add_argument('--mu', type=float, default=0.2)

    lister.add_argument('--global_model', type=str, default='ResNet18',
                        help='Type of global model: {LeNet5, CNN2, ResNet18}')
    lister.add_argument('--local_model', type=str, default='ResNet18',
                        help='Type of local model: {LeNet5, CNN2, ResNet18}')

    lister.add_argument('--dataset', type=str, default='cifar10',
                        help='datasets: {cifar100, cifar10, femnist, mnist}')
    lister.add_argument('--batchsize', type=int, default=128)
    lister.add_argument('--split', type=int, default=5)
    lister.add_argument('--val_ratio', type=float, default=0.1)
    lister.add_argument('--all_data', type=bool, default=True)
    lister.add_argument('--classes', type=int, default=10)
    lister.add_argument('--save_dir', type=str, default=None)
    lister.add_argument('--sampler', type=str, default='iid')
    lister.add_argument('--optimizer', type=str, default='sgd')
    lister.add_argument('--lr', type=float, default=0.01)
    lister.add_argument('--lr_step', type=int, default=10)
    lister.add_argument('--stop_decay', type=int, default=50)
    lister.add_argument('--momentum', type=float, default=0.9)
    lister.add_argument('--alpha', type=float, default=0.5)
    lister.add_argument('--beta', type=float, default=0.5,)

    args = lister.parse_args()
    return args
