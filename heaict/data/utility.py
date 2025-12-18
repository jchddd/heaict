import os
import numpy       as np
from pymatgen.core import Element


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


def read_mag_outcar(outcar, taks_abs=False):
    '''
    Read magnetization of each atom from OUTCAR file

    Parameter:
        - outcar (path): path to the OUTCAR file
        - take_abs (bool): Take the absolute value of the magnetic moment value extracted from OUTCAR. Default = False
    Retuen:
        - a list for magnetization of each atom
    '''
    with open(outcar, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        file.close()
    i_line_mag = []
    for i, line in enumerate(lines):
        if line == ' magnetization (x)\n':
            i_line_mag.append(i)
    for i in i_line_mag:
        if lines[i + 2] == '# of ion       s       p       d       tot\n':
            line_start = i + 4
            break
    count = 0
    for line in lines[line_start + 1:]:
        count += 1
        if line == '--------------------------------------------------\n':
            break
    line_end = line_start + count
    mag_infor = lines[line_start: line_end]
    mag_infor = [float(l.split()[4]) for l in mag_infor]
    if taks_abs:
        mag_infor = [abs(mi) for mi in mag_infor]
    
    return mag_infor
    

def df_array_process(df, columns, target_dims, sep=','):
    '''
    Turn a str of list or in DataFrame into numpy.array and reshape them for ML method

    Parameters:
        - df (pd.DataFrame): the DataFrame
        - columns (list): list of column names
        - target_dims (list): list of target dimensions corresponding to each column. reshap each arry for ML. 
        - sep (str or None): separator for data in a list. Default = ','
          If data is already array use None to cancel the transformation process of str to array. 
    '''
    for i, column in enumerate(columns):
        if sep is not None:
            df[column] = [np.fromstring(strlist[1:-1], sep=sep) for strlist in df[column]]
        else:
            df[column] = [np.array(column_value) for column_value in df[column]]
        df[column] = [array.reshape(-1, target_dims[i]) for array in df[column]]


def df_index_proceess(df, way, para):
    '''
    Function to modify indexes str for a DataFrame

    Parameters:
        - df (pd.DataFrame): the DataFrame
        - way (str): add (add suffix) or split (str split)
        - para (str): suffix for add (e.g. '.vasp') and separator for split (e.g. '.')
    '''
    if   way == 'add':
        df.index = [i + para for i in df.index]
    elif way == 'split':
        df.index = [i.split(para)[0] for i in df.index]


def check_existence(a, b):
    '''
    Function to check if vectors on a (x, n) exist on b (y, n)

    Parameters:
        - a (torch.Tensor)
        - b (torch.Tensor)
    Return:
        - 01 resut Tensor with a shape of (x, 1)
    '''
    a_expanded = a.unsqueeze(1)
    b_expanded = b.unsqueeze(0)
    comparison = a_expanded == b_expanded
    result = comparison.all(dim=-1).any(dim=-1).long().unsqueeze(1)
    return result