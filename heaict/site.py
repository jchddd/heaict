import os
import pandas                             as pd
import numpy                              as np
from collections                          import Counter
from tqdm                                 import tqdm as tqdm_common
from tqdm.notebook                        import tqdm as tqdm_notebook
from hgcode.utility                       import get_file_or_subdirection
from pymatgen.core                        import Structure, Element
from pymatgen.analysis.structure_analyzer import VoronoiConnectivity



def infer_adsorption_site(slab, adsb=['N', 'H'], adsb_excl=[], method='cutoff', cutoff=2.7, scale=1.15, max_bond=2.7, defin_dist='dist', print_neibor=False):
    '''
    Function determine the adsorption site of the molecule based on the bond length
    
    Parameters:
        - slab (pymatgen.core.Structure): the slab structure. 
        - adsb (list): species for adsorbate. Default = ['N', 'H']
        - adsb_excl (list): molecular elements that do not bond to a surface. Default = []
        - method (str): method for define bond judge distance. Default = 'cutoff'
          'cutoff': judge distance will be the cutoff value itself
          'minplus': judge distance will be the minest bond distance plus cutoff value
          'scalemin': judge distance will be the miner one between minest bond distance * scale and cutoff
        - cutoff (float): the cutoff value. Default = 0.3
        - sclae (float): the scale value. Default = 1.15
        - max_bond (float): the maximum value for bond. Default = 2.7
        - defin_dist (str): distance defination. Default = 'radius'
          'dist': bond distace will be the distance between two atoms
          'radius': bond distance will be the distance between two atoms subtract their radius
        - print_neibor (bool): whether or not to print operational information. Default = False
    Return:
        - sites: Index and element for each adsorption site atom
        - adsb_coordina: Coordina number to substrate atoms of each adsorbate atoms
        - connect: Bond indices for each atom connection. where i atoms are adsorbate atoms and j atoms are adsorption sites.
    '''
    # load element radius 
    elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Th', 'U']
    radius = [0.32, 0.31, 1.23, 0.89, 0.82, 0.77, 0.75, 0.73, 0.72, 0.71, 1.54, 1.36, 1.18, 1.11, 1.06, 1.02, 0.99, 0.98, 2.03, 1.74, 1.44, 1.32, 1.22, 1.18, 1.17, 1.17, 1.16, 1.15, 1.17, 1.25, 1.26, 1.22, 1.2, 1.17, 1.14, 1.12, 2.16, 1.91, 1.62, 1.45, 1.34, 1.3, 1.27, 1.25, 1.25, 1.28, 1.34, 1.48, 1.44, 1.4, 1.4, 1.36, 1.33, 1.31, 2.35, 1.98, 1.69, 1.65, 1.64, 1.64, 1.63, 1.62, 1.85, 1.62, 1.61, 1.6, 1.58, 1.58, 1.58, 1.74, 1.56, 1.44, 1.34, 1.3, 1.28, 1.26, 1.27, 1.3, 1.34, 1.49, 1.48, 1.47, 1.46, 1.46, 1.45, 1.65, 1.42]
    df = pd.DataFrame.from_dict({'radius': radius})
    df.index = elements
    # initial
    structure_voronoi = VoronoiConnectivity(slab)
    connectivity_array = structure_voronoi.connectivity_array
    print('sitei | sitej | distance | judge') if print_neibor else None
    adsb_index = []
    for i, specie in enumerate(slab.species):
        if specie.symbol in adsb:
            adsb_index.append(i)
    # decide bond judge distance
    sites = []
    adsb_coordina = []
    connect = [[], []]
    for i in adsb_index:
        if method == 'cutoff':
            bond_judge = cutoff
        elif method == 'scalemin' or method == 'minplus':
            distance_matrix_i = []
            for jj, dist in enumerate(slab.distance_matrix[i]):
                if slab.species[jj].symbol not in adsb:
                    if defin_dist == 'dist':
                        distance_matrix_i.append(dist)
                    elif defin_dist == 'radius':
                        distance_matrix_i.append(dist - df.at[slab[i].specie.symbol, 'radius'] - df.at[slab[jj].specie.symbol, 'radius'])
            if method == 'scalemin':
                bond_judge = min(distance_matrix_i) * scale
            elif method == 'minplus':
                bond_judge = min(distance_matrix_i) + cutoff
        if slab[i].specie.symbol in adsb_excl:
            bond_judge = -66.6
    # loop for each j and m to find adsorption site        
        coordina = 0
        for j in range(connectivity_array.shape[1]):
            for m in range(connectivity_array.shape[2]):
                if connectivity_array[i][j][m] != 0:
                    site_j = structure_voronoi.get_sitej(j, m)
                    if defin_dist == 'dist':
                        distance = site_j.distance(slab[i], jimage=[0, 0, 0])
                    elif defin_dist == 'radius':
                        distance = site_j.distance(slab[i], jimage=[0, 0, 0]) - df.at[slab[i].specie.symbol, 'radius'] - df.at[slab[j].specie.symbol, 'radius']
                    if distance < bond_judge and distance < max_bond and j not in adsb_index:
                        print('%5s | %5s | %8.5f | %5.3f' % ('-'.join([str(i), slab.species[i].symbol]), '-'.join([str(j), slab.species[j].symbol]), distance, bond_judge)) if print_neibor else None
                        connect[0].append(i)
                        connect[1].append(j)
                        sites.append('-'.join([str(j), slab.species[j].symbol]))
                        coordina += 1
        
        adsb_coordina.append(slab[i].specie.symbol + '-' + str(coordina))
            
    sites = set(sites)
    adsb_coordina = Counter(adsb_coordina)
    return sites, adsb_coordina, connect
    

def site_consistent_same_configuration(
    file_path,
    adsorbate,
    adsorption_site,
    sample=None,
    site=None,
    coordination=None,
    disable_tqdm=True,
    jupyter_tqdm=True,
    print_error_info=True,
    **para_for_infer_ads
):
    '''
    Function to check whether the adsorption sites of the same configuration are consistent under a given criterion.
    
    Parameters:
        - file_path (str, file): path to where slab files are stored
        - adsorbate (str): the name of adsorbate
        - adsorption_site (str): the letter representing the adsorption site.
        - sample (str): the slab file name in which the adsorption site can be correctly identified under the given criterion. Default = None
        - site (dict): the correct site from infer_adsorption_site. Default = None
        - coordination (Counter): the correct coordination from infer_adsorption_site. Default = None 
        - disable_tqdm (bool). Default = False
        - jupyter_tqdm (bool). Default = True
        - print_error_info (bool): whether to print the error site and coordination. Default = True
    Return:
        - A list of error slab file names.
    Cautions：
        - This function is to check the site consistency under different slab with a same configuration
        - You should give a sample or give the correct site and coordination at the same time
    Example:
        site_consistent_same_configuration(r'8ele_fcc_data\FPS_ori','N2v','t','200_N2v_0_0t.vasp')
    '''
    assert (site is not None and coordinate is not None) or sample is not None, 'you should give one correct example !'
    print('Check the site determination of slabs with same adsorption configuration')
    print('Expected result:')
    error_slabs = []
    # get correct site and coordinate
    coor_to_site = {4: '4-fold', 3: 'hollow', 2: 'bridge', 1: 'top'}
    if sample is not None:
        slab = Structure.from_file(os.path.join(file_path, sample))
        site_correct, coordination_correct, connection = infer_adsorption_site(slab, **para_for_infer_ads)
    else:
        site_correct = site
        coordination_correct = coordination
    print('    site: ', coor_to_site.get(len(site_correct), 'other'), site_correct, )
    print('    coor: ', coordination_correct)
    # init counter
    number_check = 0
    number_error = 0
    # loop for each slab and check site consistency
    slabs = get_file_or_subdirection(file_path)
    tqdm = tqdm_notebook if jupyter_tqdm else tqdm_common
    with tqdm(slabs, disable=disable_tqdm) as pbar:
        for slab_file in pbar:
            if '_' + adsorbate + '_' in slab_file and slab_file.split('.')[0][-1] == adsorption_site:
                number_check += 1
                slab = Structure.from_file(os.path.join(file_path, slab_file))
                site, coordination, connection = infer_adsorption_site(slab, **para_for_infer_ads)
                if len(site) != len(site_correct) or coordination != coordination_correct:
                    number_error += 1
                    error_slabs.append(slab_file)
                    if print_error_info:
                        print('Error ', str(number_error), slab_file)
                        print('    site: ', coor_to_site.get(len(site), 'other'), site)
                        print('    coor: ', coordination)
    print('error/total: ', str(int(number_error)), '/', str(int(number_check)))
    
    return error_slabs    


def site_consistent_same_slab(
    file_paths,
    parameter_dicts,
    disable_tqdm=True,
    jupyter_tqdm=True,
    print_error_info=True,
):
    '''
    Function to check whether adsorption sites of a same slab are consistent after different treatments to the slab with different criterion.
    
    Parameters:
        - file_paths (list): file paths to slabs with different treatments
        - parameter_dicts (list): dicts of key value pairs of parameter on infer_adsorption_site
        - disable_tqdm (bool). Default = False
        - jupyter_tqdm (bool). Default = True
        - print_error_info (bool). Default = True
    Return:
        - A list of error slab file names.
    Cautions：
        - File paths and parameter dicts have to correspond one to one.
    Example:
        site_consistent_same_slab(
            file_paths=[
                r'data_mlr', 
                r'data_ori'],
            parameter_dicts=[
                {'cutoff': 0.36, 'defin_dist': 'radius', 'adsb_excl': ['H']},
                {'cutoff': 2.7, 'defin_dist': 'dist'},],)        
    '''
    print('Check whether adsorption sites of a same slab change after different treatments')
    coor_to_site = {4: '4-fold', 3: 'hollow', 2: 'bridge', 1: 'top'}
    number_slab_type = len(file_paths)
    error_slabs = []
    # init counter
    number_check = 0
    number_error = 0
    # prepare loop 
    slab_files = get_file_or_subdirection(file_paths[0])
    tqdm = tqdm_notebook if jupyter_tqdm else tqdm_common
    with tqdm(slab_files, disable=disable_tqdm) as pbar:
        # loop for each slab
        for slab_file in pbar:
            number_check += 1
            sites = []
            coordinations = []
            # loop for each slab type
            for i in range(number_slab_type):
                slab = Structure.from_file(os.path.join(file_paths[i], slab_file))
                site, coor, cnet = infer_adsorption_site(slab, **parameter_dicts[i])
                sites.append(site)
                coordinations.append(coor)
            sites_str = [coor_to_site.get(len(site), 'other') for site in sites]
            # check consistency
            site_check = []
            coordination_check = []
            for j in range(number_slab_type - 1):
                site_check.append(sites_str[j] == sites_str[j + 1])
                coordination_check.append(coordinations[j] == coordinations[j + 1])
            if not (all(site_check) and all(coordination_check)):
                number_error += 1
                error_slabs.append(slab_file)
                if print_error_info:
                    print('error ', str(number_error), slab_file)
                    print(' '.join(sites_str))
                    print(' '.join([str(dict(coordination)) for coordination in coordinations]))
    print('error/total: ', str(int(number_error)), '/', str(int(number_check)))
    
    return error_slabs
    
    
def infer_molecular_angle(slab, adsb_skele=['N'], print_info=False):
    '''
    Function to determine the adsorption form of the molecule according to the angle of its skeleton
    
    Parameters:
        - slab (pymatgen.core.Structure): the slab structure.
        - adsb_skele (list): adsorbed molecular skeleton elements. Default = ['N']
        - print_info (bool): whether to print operational information or not. Default = False
    Return:
        - A single word that represent the molecular angle, like 'v' (vertical), 'h' (horizontal), and 'i' (tilted).
    Cautions：
        - It can only be used when the skeleton contains only two atoms.
    '''
    skele_coords = []
    for i in range(len(slab)):
        if slab[i].specie.symbol in adsb_skele:
            skele_coords.append(slab.cart_coords[i])
            
    v_skele = np.array(skele_coords[1] - skele_coords[0])
    length = np.linalg.norm(v_skele)
    z_component = np.abs(v_skele[2])
    angle_rad = np.arcsin(z_component/length)
    angle_deg = np.degrees(angle_rad)
    
    angle_diff = [np.abs(90-angle_deg), np.abs(0-angle_deg), np.abs(45-angle_deg)]
    min_diff_index = angle_diff.index(min(angle_diff))
    
    if print_info:
        print('vector: ', v_skele)
        print('angle: ', angle_deg)
    
    if min_diff_index == 0:
        return 'v' # vertical
    elif min_diff_index == 1:
        return 'h' # horizontal
    elif min_diff_index == 2:
        return 'i' # tilted 