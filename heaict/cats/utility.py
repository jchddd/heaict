import numpy as np
import os
import inspect
from collections import Counter

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

def sort_by_frequency_unique(vec):
    '''
    Input a vector and return a set of independent elements based on the frequency of occurrence of the elements within it from high to low

    Parameters:
        - vec: the vector
    Return:
        - list of unique element based on the frequency of occurrence
    '''
    freq_counter = Counter(vec)
    sorted_elements = sorted(freq_counter.keys(), key=lambda x: (-freq_counter[x]))
    return sorted_elements

def round_preserve_sum(x, decimals=2):
    '''
    Perform a round operation on the numbers within a vector and ensure that the sum after round is 1.

    Paremeters:
        - x: the cevtor
        - decimals (int): the number of decimal places retained. Default=2
    Return:
        - A rounded vector whose sum equals to 1.
    '''
    x_rounded = np.round(x, decimals)
    error = 1.0 - x_rounded.sum()
    while not np.isclose(error, 0):
        diffs = x - x_rounded
        if error > 0:
            idx = np.argmax(diffs)
        else:
            idx = np.argmin(diffs)
        adjustment = error
        x_rounded[idx] += adjustment
        x_rounded = np.round(x_rounded, decimals)
        error = 1.0 - x_rounded.sum()
    return x_rounded

def sort_by_columns_lexicographically(arr):
    '''
    Lexicographically sorts the rows of an n x m dimensional array according to the value of the column
    
    Paremater:
        - arr (np.array with a shape of (n, m))
    Return:
        - The sorted array
    '''
    sorted_indices = np.lexsort([arr[:, i] for i in range(arr.shape[1])][::-1])
    sorted_arr = arr[sorted_indices]
    
    return sorted_arr