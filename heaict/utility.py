import os
import inspect
import numpy       as np
from pymatgen.core import Element


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


def get_element_symbol(data, asedb=True):
    '''
    Obtain the unique elements from ase database or a list of graphs
    
    Parameters:
        - data (list): list of graphs (torch_geometric.data.Data).
        - asedb (bool): whether data is ase database or torch graph list. Default = True
    Return:
        - The list of unique elements.
    '''
    if asedb:
        Zs = [z for z in np.unique(np.concatenate([row.toatoms().get_atomic_numbers() for row in data.select()]))]
    else:
        Zs = [z.item() for z in torch.cat([graph.atomic_number.long() for graph in graphs], dim=0).unique()]
    return [Element.from_Z(z).symbol for z in Zs]
    
    
def get_file_or_subdirection(path, type='file'):
    '''
    Function that gets all subdirectories or files in a specified directory

    Parameters:
        - path: Directory where the command is executed / str(path)
        - type: Return files or subdirectories / str, 'file' or 'dir', default 'file'
    Return:
        - List of file or direction names
    '''
    for curDir, dirs, files in os.walk(path):
        if curDir == path and type == 'file':
            return files
        elif curDir == path and type == 'dir':
            return dirs


def find_indices(lst, value):
    '''
    Function to find indices for values from a list.
    
    Parameters:
        - lst (array like).
        - value (array like).
    Return:
        - The indices (np.array).
    '''
    return np.array([i for i, v in enumerate(lst) if v in value])


def get_function_parameter_name(function):
    '''
    Function to get parameter names of a function
    
    Parameter:
        - function (function object).
    Return:
        - The list of parameter names.
    '''
    parameter_names = []
    for name, parameter in inspect.signature(function).parameters.items():
        parameter_names.append(name)
    return parameter_names

def create_reverse_dict(original_dict):
    '''
    Function to create a reverse dict with key and value reversed, list values will be unzip to keys.
    
    Parameter:
        - original_dict (dict).
    Return:
        - The reverse dict.
    '''
    reverse_dict = {}
    for key, values in original_dict.items():
        for value in values:
            reverse_dict[value] = key
    return reverse_dict
    
def check_array1_contained_in_array2(array1, array2, check_way='all'):
    '''
    Function to check if vectors in array1 are all or any contained in array2.
    
    Parameters:
        - array1 (numpy.array)
        - array2 (numpy.array)
        - check_way ('all' or 'any')
    Return:
        - True if array1 is all or any contained in array2, else False
    Cautions:
        - array1 and array2 should be shape of (any_number, same_number)
    '''
    item1_in_array2 = []
    for item1 in array1:
        item1_in_array2.append(np.any([np.array_equal(item1, item2) for item2 in array2]))
    if getattr(np, check_way)(item1_in_array2):
        return True
    else:
        return False