from pymatgen.io.vasp import Poscar
from pymatgen.core    import Structure, Lattice
import pandas         as pd
import numpy          as np
import random
import os


def slab_pure_to_hea(origin_slab, origin_ele, eleAlat, num_ele=[5], num_hea=10, save_path=f'HEA_slabs', step=0.01, 
                     constraint=[], adsb_ele=['N', 'H'], vacuum=15, surface_adjusted=True, save_by_index=True, seed=None):
    '''
    Creating HEA slabs by randomly replacing the atoms of the pure element slab

    Parameters:
        - origin_slab (pymatgen.core.Structure): the origin pure slab
        - origin_ele (str): the metallic element on the origin pure slab
        - eleAlat (dict): element names as keys and bulk lattice as values
        - num_ele (list of int): number of metallic elements on HEA slab. Default = [5]
        - num_hea (int): number of generated HEA slabs. Default =10
        - save_path (str of path): direction where to save the HEA slab structure files. Default = f'HEA_slabs'
        - step (float): step of atomic fraction. Should be able to be divided exactly by 1. Default = 0.01
        - constraint (list of (3) tuples): list of constraints, including element, symbol, and fraction. Default = []
          e.g. [('Fe', 'gt', 0.3)] means HEA must contain Fe with atomic fraction >= 0.3. symbol can choose from 'gt', 'lt'
        - adsb_ele (list): list of adsorbate elements. Default = ['N', 'H']
        - vacuum (flaot): vacuum layer height. Default = 15
        - surface_adjusted (bool): whether to adjust lattice according to the topmost surface atoms. Default = True
        - save_by_index (bool): save files by generated indices, if not save by element and fraction. Default = True
        - seed (None or int): random seed in numpy. Default = None
    Return:
        - A DataFrame with index and HEA fraction
    '''
    # inits
    if seed is not None:
        np.random.seed(seed)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    elements = list(eleAlat.keys())
    ele_index=np.arange(len(elements))
    output = {'index':[],'HEA':[]}
    # loop for number of hea slabs
    for i in range(num_hea):
        while True:
            # get random elements and their atomic fraction
            ele_num = np.random.choice(num_ele)
            np.random.shuffle(ele_index)
            rnum=generate_random_numbers(ele_num, 1, step)
            p={}
            rni=0
            for ei in ele_index[:ele_num]:
                p[elements[ei]]=rnum[rni]
                rni+=1
            # constraint
            if   len(constraint) == 0:
                break
            else:
                g = []
                for constra in constraint:
                    if   constra[1] == 'gt':
                        if constra[0] in p.keys() and p[constra[0]] >= constra[2]:
                            g.append(True)
                        else:
                            g.append(False)
                    elif constra[1] == 'lt':
                        if constra[0] in p.keys() and p[constra[0]] <= constra[2]:
                            g.append(True)
                        else:
                            g.append(False)
                    else:
                        g.append(True)
                if np.all(g):
                    break
        # substitution
        prot_struc = origin_slab.copy()
        slab_struc = substitute_in_probability(prot_struc, {origin_ele: p}, 'ration')
        slab_struc = average_lattice(slab_struc,
                                     eleAlat, 
                                     eleAlat[origin_ele],
                                     vacuum,
                                     True,
                                     adsb_ele)
        # save 
        HEA = '_'.join([ele + '-' +str(int(round(p[ele] * 100, 0))) for ele in p.keys()])
        if save_by_index:
            save_structure(slab_struc, os.path.join(save_path, str(i+1) + '.vasp'))
        else:
            save_structure(slab_struc, os.path.join(save_path, HEA + '.vasp'))
        output['index'].append(i+1)
        output['HEA'].append(HEA)
        
    df=pd.DataFrame.from_dict(output)
    return df

def substitute_in_probability(struc, sub_pro, method='probability'):
    '''
    Replace atomic elements in the structure according to their proportions

    Parameters:
        - struc (pymatgen.core.Structure)
        - sub_pro (dict of dicts): The element to be replaced, and each element used for replacement and its proportion.
          e.g. {'Cu': {'Fe': 0.5, 'Ru': 0.5}} means replace Cu with Fe and Ru. The Sum of proportions must be 1.
        - method (str): 'probability' or 'ration'. Default = 'probability'
    Return:
        - The structure after replacing the elements
    '''
    s = struc.copy()
    
    if method == 'probability':
        for i, ele in enumerate([atom.symbol for atom in s.species]):
            if ele in sub_pro.keys():
                p = list(sub_pro[ele].values())
                sub_ele = np.random.choice(list(sub_pro[ele].keys()), p=p)
                s[i] = sub_ele
                
    elif method == 'ration':
        bsub_atom_list = []
        sub_ele_list = []

        for i_bse, bsub_ele in enumerate(list(sub_pro.keys())):
            bsub_atom_list.append([])
            sub_ele_list.append([])

            bsub_atom_count = 0
            for i_atom, ele in enumerate(s.species):
                if ele.symbol == bsub_ele:
                    bsub_atom_count += 1
                    bsub_atom_list[i_bse].append(i_atom)

            sub_ele_num = [int(bsub_atom_count * sub_pro_ele) for sub_pro_ele in list(sub_pro[bsub_ele].values())]
            sub_ele_num[-1] = bsub_atom_count - sum(sub_ele_num[:-1])
            for i_sube, sub_ele in enumerate(list(sub_pro[bsub_ele].keys())):
                for sub_num in range(sub_ele_num[i_sube]):
                    sub_ele_list[i_bse].append(sub_ele)

        for ij, ll in enumerate(bsub_atom_list):
            random.shuffle(sub_ele_list[ij])
            for ik, iatom in enumerate(ll):
                s[iatom] = sub_ele_list[ij][ik]
                
    return s

def decimal_places(num):
    num_str = str(num)
    decimal_index = num_str.find('.')
    
    if decimal_index == -1:
        return 0
    else:
        return len(num_str) - decimal_index - 1

def generate_random_numbers(number=5, sumv=1, step=0.05, method=0):
    '''
    Generate a set of random numbers with fixed step size and sum

    Parameters:
        - number (int): the total number of random numbers. Default = 5
        - sumv (int): sum of random numbers. Default = 1
        - step (float): step size. Default = 0.05
        - method (int): 0: generate a group of random numbers and normalize them
          Default = 0   1: generate a random number each time from the remaining intervals
    Return:
        - list of random numbers
    '''
    if method == 0:
        random_number = np.random.rand(number)
        total = np.sum(random_number)
        result = [num / total * sumv for num in random_number]
        
    elif method == 1:
        result = []
        total = sumv
        for _ in range(number - 1):
            result.append(np.random.uniform(0, total))
            total -= result[-1]
        result.append(sumv - np.sum(result))
            
    result = [np.int32(np.round(num / step)) for num in result]
    total_step = np.int32(sumv / step)
    result[-1] = np.int32(total_step - np.sum(result[: -1]))
    if result[-1] < 0:
        result[-1] = np.int32(0)
        result[-2] = np.int32(total_step - np.sum(result[: -2]))
    result = [np.round(step_num * step, decimal_places(step)) for step_num in result]
            
    return result

def average_lattice(struc, data_lat, origin_lattice, vacuum=15, surface_adjusted=True, adsb_species=['N', 'H']):
    '''
    Scale the structure based on the average lattice constant

    Parameters:
        - struc (pymatgen.core.Structure): single-metal element slab structure
        - data_lat (dict): The str of elements are keys, and the lattice constants of bulk are values
        - origin_lattice (float): the bulk lattice constant of the origin single-metal
        - vacuum (float): height of the vacuum layer. Default = 15
        - surface_adjusted (bool): adjust the lattice constant in the horizontal direction (x, y) 
          based on the average lattice constant of the topmost layer elements. Default = True
        - adsb_species (list): list of adsorbate elements. Default = ['N', 'H']
    Return:
        - The structure after averaging the lattice constant
    '''
    total_radius = 0
    number_atom = 0
    for spec in struc.species:
        if spec.symbol in data_lat.keys():
            total_radius += data_lat[spec.symbol]
            number_atom += 1
    average_radius = total_radius / number_atom
    scale = average_radius / origin_lattice

    lattice_scale = Lattice.from_parameters(struc.lattice.a * scale, struc.lattice.b * scale, struc.lattice.c * scale, 
                                            struc.lattice.alpha, struc.lattice.beta, struc.lattice.gamma)
    struc_scale = Structure(lattice_scale, struc.species, struc.frac_coords)

    lattice_scale_redfz = Lattice.from_parameters(struc.lattice.a * scale, struc.lattice.b * scale, (struc.lattice.c - vacuum) * scale + vacuum, 
                                            struc.lattice.alpha, struc.lattice.beta, struc.lattice.gamma)
    struc_scale_redfz = Structure(lattice_scale_redfz, struc.species, struc_scale.cart_coords, coords_are_cartesian=True)

    if surface_adjusted:
        LDP = Layer_Divide_Process()
        LDP.print_info = False
        struc_scale_rmads = struc_scale_redfz.copy()
        struc_scale_rmads.remove_species(adsb_species)
        LDP.load_slab(struc_scale_rmads)
        LDP.divide_layer()
        surface_atoms = LDP.identify_layer([LDP.layer_number - 1], 'layer')

        total_radius = 0
        number_atom = 0
        for surface_atom in surface_atoms:
            atom_symbol = struc_scale_rmads[surface_atom].specie.symbol
            if atom_symbol in data_lat.keys():
                total_radius += data_lat[atom_symbol]
                number_atom += 1
        average_radius_surface = total_radius / number_atom
        scale_surface = average_radius_surface / (origin_lattice * scale)

        lattice_scale_surf = Lattice.from_parameters(struc_scale_redfz.lattice.a * scale_surface, 
                                                     struc_scale_redfz.lattice.b * scale_surface, 
                                                     struc_scale_redfz.lattice.c, 
                                        struc.lattice.alpha, struc.lattice.beta, struc.lattice.gamma)
        struc_scale_surf = Structure(lattice_scale_surf, struc.species, struc_scale_redfz.frac_coords)
    
    if surface_adjusted:
        struc_final = struc_scale_surf
    else:
        struc_final = struc_scale_redfz

    if 'selective_dynamics' in struc.site_properties.keys():
        struc_final.add_site_property('selective_dynamics', struc.site_properties['selective_dynamics'])
        
    return struc_final

def save_structure(structure, file):
    '''
    Function to save structure to POSCAR file with sort species

    Parameter:
        - structure: Structure that need to write to a POSCAR / pymatgen.core.Structure
        - file: File that the structure write to / str(path)
    '''
    poscar = Poscar(structure, sort_structure=True)
    poscar.write_file(file)

class Layer_Divide_Process():
    '''
    Class for layering structures and fixing or moving atoms for each layer individually

    Available Functions:
        - load_slab: Function to load slab structure
        - divide_layer: Functions that perform layered operations on the slab
        - identify_layer: Function for Determining atomic layers or obtaining atoms of a specific layer
        - fix_layer: Function to fix coordinates according to the atomic layer
        - move_layer: Function for moving atoms for specific layers
        - delete_layer: Function to delete layer atoms
        - rotate_layer: Function to rotate layers
        - reset_structure: Function to reset the processed structure to inital structure
        - save_structure: Function to save the processed structure to POSCAR file
        - view_structure: Function to view the processed structure
    '''

    def __init__(self):
        '''
        Attributes:
            - work_path: Path to where store the output files / str,path
            - structure_init: The initial structure / pymatgen.core.Structure
            - structure_process: Structures after fixed or mobile processing / pymatgen.core.Structure
            - devide_method: Names of layering methods and their key parameters / (2) list, [method name, the key parameter]
            - layer_refer: reference z coordinates for help with layering / (l) list, l is the number of layers
            - layer_number: Total number of layer according to the devide mode / int
            - layer_list: Layer for each atom. Note that the layer start from 0 / (n) list, n is the atom number in slab
            - layer_bar: Histogram data that counts the number of atoms in each layer / (l) list, l is the number of layers
            - print_info: Whether to pring running information / bool, default True
        '''
        self.work_path = None
        self.structure_init = None
        self.structure_process = None
        self.devide_method = []
        self.layer_refer = []
        self.layer_number = 0
        self.layer_list = []
        self.layer_bar = []

        self.print_info = True

    def load_slab(self, structure):
        '''
        Function to load slab structure

        Parameter:
            - structure: Input slab structure / path or pymatgen.Structure
        Accomplish:
            - Read structure to structure_init and copy it to structure_process
        '''
        # read structure and set work path
        if type(structure) == type('a'):
            self.structure_init = Structure.from_file(structure)
            self.work_path = os.path.split(os.path.abspath(structure))[0]
        else:
            self.structure_init = structure
            self.work_path = os.getcwd()
        self.structure_process = self.structure_init.copy()
        # print info
        if self.print_info:
            utility.screen_print('Load Slab')
            utility.screen_print('Load structure  ', 'Compeleted')
            utility.screen_print('Slab atom number', str(len(self.structure_init)))
            utility.screen_print('End')

    def divide_layer(self, identify_method='threshold', method_parameter=0.36):
        '''
        Functions that perform layered operations on the slab

        Parameters:
            - identify_method: Method to devide atomic layer / str in 'round' or 'threshold', default 'threshold'
                The 'round' method uses round() to take an approximation of the z coordinates. After the
                approximation, atoms with the same coordinates are considered to be at the same layer.
                The 'threshold' will search refer z In sequence. Atoms with z closes to one specific refer
                z with a difference less than a threshold value will be considerd to be at the same layer.
            - method_parameter: method parameter / int or float, default 0.36
                In 'round' method, it is the int parameter in round(), like 2 or 3
                In 'threshold', it is the threshold value, a value like 0.3 is rational
        '''
        # build layer refer
        self.devide_method = [identify_method, method_parameter]
        # round method
        if identify_method == 'round':
            atomic_zs_frac = [round(coord[2], method_parameter) for coord in self.structure_init.frac_coords]
            unique_z_frac = list(set(atomic_zs_frac))
            unique_z_frac.sort()
            self.layer_refer = unique_z_frac
        # threshold method
        elif identify_method == 'threshold':
            atomic_zs_cart = [coord[2] for coord in self.structure_init.cart_coords]
            refer_z_cart = [atomic_zs_cart[0]]
            for atomic_z in atomic_zs_cart:
                to_refer = True
                for refer_z in refer_z_cart:
                    if abs(atomic_z - refer_z) <= method_parameter:
                        to_refer = False
                if to_refer:
                    refer_z_cart.append(atomic_z)
            refer_z_cart.sort()
            self.layer_refer = refer_z_cart
        # collect info
        self.layer_number = len(self.layer_refer)
        self.layer_bar = [0] * self.layer_number
        self.layer_list = self.identify_layer(range(len(self.structure_init)))
        for atom_index in range(len(self.structure_init)):
            self.layer_bar[self.layer_list[atom_index]] += 1
        # print info
        if self.print_info:
            utility.screen_print('Devide Layer')
            utility.screen_print('Layer number', str(self.layer_number))
            utility.screen_print('Layer refer ', [round(i, 2) for i in self.layer_refer])
            utility.screen_print('Layer bar   ', self.layer_bar)
            utility.screen_print('End')

    def identify_layer(self, identify_list, by='atom'):
        '''
        Function for Determining atomic layers or obtaining atoms of a specific layer

        Parameters:
            - identify_list: List of data that need to used at identified layers / (n) list
            - by: Input atom indexes to identify layers or input layers to find atoms / str, 'atom' or 'layer', default 'atom'
        Return:
            - 'atom', return a (n) list of layers corresponding to each atoms
            - 'layer', return a (n) list of atoms that locate on these layers
        '''
        # init
        identiyf_list = []
        # identify by atom
        if by == 'atom':
            for atom_index in identify_list:
                if self.devide_method[0] == 'round':
                    atomic_z = round(self.structure_init.frac_coords[atom_index][2], self.devide_method[1])
                    layer = self.layer_refer.index(atomic_z)
                    identiyf_list.append(layer)
                elif self.devide_method[0] == 'threshold':
                    atomic_z = self.structure_init.cart_coords[atom_index][2]
                    for refer_z in self.layer_refer:
                        if abs(atomic_z - refer_z) <= self.devide_method[1]:
                            layer = self.layer_refer.index(refer_z)
                            break
                    identiyf_list.append(layer)
        # identify by layer
        elif by == 'layer':
            for atom_index, atomic_layer in enumerate(self.layer_list):
                if atomic_layer in identify_list:
                    identiyf_list.append(atom_index)
        # return
        return identiyf_list