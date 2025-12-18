from scipy.stats import norm
from colorama import Fore
import numpy as np
import random
import torch
import os
import time


def fix_seed(seed):
    '''
    Function used to set the random seed

    Args:
        - seed: The random seed / int
    '''
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_max_num_nodes(graphs):
    '''
    Obtain the maximum number of nodes in a series of graphs
    
    Parameter:
        - graphs (list): list of graphs (torch_geometric.Data).
    Return:
        - The maximum number of nodes.
    '''
    max_nodes = 0
    for graph in graphs:
        max_nodes = max(max_nodes, graph.num_nodes)
    return max_nodes

def count_parameters(item, print_info=False):
    if print_info:
        for name, module in item.named_children():
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"{name}: Number of trainable parameters: {params}")
            trainable_variables = [param for param in module.parameters() if param.requires_grad]
            for var_name, var_param in module.named_parameters():
                if var_param.requires_grad:
                    print(f"  - {var_name} (size: {var_param.size()})")
    return sum(p.numel() for p in item.parameters())

def set_requires_grad(item, requires_grad):
    for param in item.parameters():
        param.requires_grad = requires_grad

def safe_divide(x1, x2):
    '''
    Divide x1 / x2, return 0 where x2 == 0
    '''
    return np.divide(x1, x2, out=np.zeros(np.broadcast(x1, x2).shape), where=(x2 != 0))

def get_z_value(confidence_level):
    '''
    get z value according to confidence_level
    '''
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must locate at 0 to 1")

    z_value = norm.ppf(1 - (1 - confidence_level) / 2)

    return z_value

def print_with_timestamp(message, color='BLACK'):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(getattr(Fore, color) + f"{message} - [{timestamp}]")