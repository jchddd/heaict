import os
import pandas                             as pd
import numpy                              as np
from collections                          import Counter
from tqdm                                 import tqdm as tqdm_common
from tqdm.notebook                        import tqdm as tqdm_notebook
from heaict.data.utility                  import get_file_or_subdirection
from pymatgen.core                        import Structure, Element
from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
from ase.data                             import covalent_radii
from pymatgen.io.ase                      import AseAtomsAdaptor
from ase                                  import neighborlist
from ase.neighborlist                     import natural_cutoffs
from shutil                               import copy, move


def infer_adsorption_site(slab, adsb=['N', 'H'], method='cutoff', judge=2.7, max_bond=2.7, defin_dist='dist', pymr=True, print_neibor=False):
    '''
    Function determine the adsorption site of the molecule based on the bond length
    
    Parameters:
        - slab (pymatgen.core.Structure): the slab structure. 
        - adsb (list): species for adsorbate. Default = ['N', 'H']
        - method (str): method for define bond judge distance. Default = 'cutoff'
          A bond whose length is less than the judge distance is considered a bond.
          'cutoff': judge distance will be the judge value itself
          'minplus': judge distance will be the minest bond distance for each adsorbate atom plus judge value
          'scalemin': judge distance will be the minest bond distance for each adsorbate atom multiply by judge
          'scaleradius': judge distance will be sum of atom radius multipy by judge. defin_dist is not work for this method.
        - judge (float): value used to help defend the judge distance. Default = 2.7
        - max_bond (float): the maximum value for bond defined by 'dist'. Default = 2.7
        - defin_dist (str): distance defination. Default = 'dist'
          'dist': bond distace will be the distance between two atoms
          'radius': bond distance will be the distance between two atoms subtract their radius
        - pymr (bool): use the radius from pymatgen, if not use radius from ase. Default = True
        - print_neibor (bool): whether or not to print operational information. Default = False
    Return:
        - sites: Index and element for each adsorption site atom
        - adsb_coordina: Coordina number to substrate atoms of each adsorbate atoms
        - connect: Bond indices for each atom connection. where i atoms are adsorbate atoms and j atoms are adsorption sites.
    '''
    # get distance matrix for adsorbate 
    adsb_indices = np.where(np.logical_or.reduce([np.array(slab.atomic_numbers) == value for value in [Element(e).number for e in adsb]]))[0]
    adsb_dist_matrix = slab.distance_matrix[adsb_indices]
    # mask self and adsb
    adsb_dist_matrix = np.where(adsb_dist_matrix !=0, adsb_dist_matrix, 6.66666)
    adsb_dist_matrix[:, adsb_indices] = 6.66666
    # calculate min and max ditance matrix
    if   method == 'cutoff' and defin_dist == 'dist':
        matrix_minmum = np.full(adsb_dist_matrix.shape, 0)
        matrix_maxmum = np.full(adsb_dist_matrix.shape, judge)
    elif method == 'minplus' and defin_dist == 'dist':
        matrix_minmum = np.repeat(np.min(adsb_dist_matrix, axis=1).reshape(-1, 1), adsb_dist_matrix.shape[1], axis=1)
        matrix_maxmum = matrix_minmum + judge
    elif method == 'scalemin' and defin_dist == 'dist':
        matrix_minmum = np.repeat(np.min(adsb_dist_matrix, axis=1).reshape(-1, 1), adsb_dist_matrix.shape[1], axis=1)
        matrix_maxmum = matrix_minmum * judge
    elif method == 'cutoff' and defin_dist == 'radius':
        array_radius_target = np.array([site.specie.atomic_radius if pymr else covalent_radii[site.specie.number] for site in slab])
        array_radius_source = np.concatenate([np.full((1, len(slab)), slab[ai].specie.atomic_radius if pymr else covalent_radii[slab[ai].specie.number]) for ai in adsb_indices], axis=0)
        array_radius = array_radius_target + array_radius_source
        matrix_minmum = array_radius
        matrix_maxmum = array_radius + judge
    elif method == 'minplus' and defin_dist == 'radius':
        array_radius_target = np.array([site.specie.atomic_radius if pymr else covalent_radii[site.specie.number] for site in slab])
        array_radius_source = np.concatenate([np.full((1, len(slab)), slab[ai].specie.atomic_radius if pymr else covalent_radii[slab[ai].specie.number]) for ai in adsb_indices], axis=0)
        array_radius = array_radius_target + array_radius_source
        matrix_minmum = array_radius + np.repeat(np.min(adsb_dist_matrix - array_radius, axis=1).reshape(-1, 1), adsb_dist_matrix.shape[1], axis=1)
        matrix_maxmum = matrix_minmum + judge
    elif method == 'scalemin' and defin_dist == 'radius':
        array_radius_target = np.array([site.specie.atomic_radius if pymr else covalent_radii[site.specie.number] for site in slab])
        array_radius_source = np.concatenate([np.full((1, len(slab)), slab[ai].specie.atomic_radius if pymr else covalent_radii[slab[ai].specie.number]) for ai in adsb_indices], axis=0)
        array_radius = array_radius_target + array_radius_source
        mindist_matrix = np.repeat(np.min(adsb_dist_matrix - array_radius, axis=1).reshape(-1, 1), adsb_dist_matrix.shape[1], axis=1)
        matrix_minmum = array_radius + mindist_matrix
        matrix_maxmum = array_radius + mindist_matrix * judge
    elif method == 'scaleradius':
        array_radius_target = np.array([site.specie.atomic_radius if pymr else covalent_radii[site.specie.number] for site in slab])
        array_radius_source = np.concatenate([np.full((1, len(slab)), slab[ai].specie.atomic_radius if pymr else covalent_radii[slab[ai].specie.number]) for ai in adsb_indices], axis=0)
        array_radius = array_radius_target + array_radius_source
        matrix_minmum = array_radius
        matrix_maxmum = array_radius * judge
    # get bond indices
    # bond_adsb, bond_metal = np.where((matrix_minmum <= adsb_dist_matrix) & (adsb_dist_matrix < matrix_maxmum) & (adsb_dist_matrix < max_bond))
    bond_adsb, bond_metal = np.where((adsb_dist_matrix <= matrix_maxmum) & (adsb_dist_matrix <= max_bond))
    not_adsb = np.where(~np.in1d(bond_metal, adsb_indices))[0]
    bond_adsb = adsb_indices[bond_adsb[not_adsb]]
    bond_metal = bond_metal[not_adsb]
    # get output
    site = set([str(bmi) + '-' +slab[bmi].specie.symbol for bmi in bond_metal])
    adsb_coordinate = Counter([slab[ai].specie.symbol + '-' + str(np.count_nonzero(bond_adsb == ai)) for ai in adsb_indices])
    connect = [list(bond_adsb), list(bond_metal)]
    # print
    if print_neibor:
        print('sitei | sitej | distance | judge')
        for i, j in zip(*connect):
            ai = np.where(adsb_indices == i)[0][0]
            if   defin_dist == 'dist':
                print('%5s | %5s | %8.5f | %5.3f' % ('-'.join([str(i), slab.species[i].symbol]), '-'.join([str(j), slab.species[j].symbol]), adsb_dist_matrix[ai][j], np.where(matrix_maxmum < max_bond, matrix_maxmum, max_bond)[ai][j]))
            elif defin_dist == 'radius':
                print('%5s | %5s | %8.5f | %5.3f' % ('-'.join([str(i), slab.species[i].symbol]), '-'.join([str(j), slab.species[j].symbol]), (adsb_dist_matrix - array_radius)[ai][j], (np.where(matrix_maxmum < max_bond, matrix_maxmum, max_bond)- array_radius)[ai][j]))
    return site, adsb_coordinate, connect


def infer_adsorption_site_voronoi(slab, adsb=['N', 'H'], adsb_excl=[], method='cutoff', cutoff=2.7, scale=1.15, max_bond=2.7, defin_dist='dist', print_neibor=False):
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
    

def site_process(
    path_all_ori,
    path_ori,
    path_opt,
    para_infer_site_origin={},
    para_infer_site_optima=[{}],
    isomorphism_ori=[],
    isomorphism_opt=[],
    cutoff_multiplier=1.3,
    adsb_symbol=['N', 'H'],
    uniform_adsb={'N2h': 'N2', 'N2v': 'N2', 'NNHh': 'NNH', 'NNHv': 'NNH', 'NH': 'NH', 'NH3': 'NH3', 'H': 'H'},
    deal_file=False,
    error_path=None,
    dataframe=None,
    detect_anomaly=True,
    disable_tqdm=True,
    jupyter_tqdm=True,
):
    '''
    A process that combines anomaly detection, site change detection, and finding initial configurations.

    Parameters:
        - path_all_ori (path): path to all constructed initial slab structures.
        - path_ori (path): path to calculated initial slab structures.
        - path_opt (path): path to relaxed slab structures.
        - para_infer_site_origin (dict): a dict of key value pairs of parameter on infer_adsorption_site for origin slab. Default = {}
        - para_infer_site_optima (list of dict): a list of dicts of key value pairs of parameter on infer_adsorption_site for relaxed slab. Default = [{}]
        - isomorphism_ori (list of dict): dicts of coordination. If an original slab has those coordination, then its optimized structure may have an 
          isomorphism. This corresponds to the coordination in isomorphism_opt.
        - isomorphism_opt (list of dict): dicts of coordination. Suppose a slab has this coordination, and the site is the same, and the connection
          relationship contains the original. In that case, this slab will also be considered the same configuration to the origin slab. Default = None.
        - cutoff_multiplier (float): the radius multiple used for the criterion. Default = 1.3
        - adsb_symbol (list): symbol for adsorbate. Default = ['N', 'H']
        - uniform_adsb (dict): convert molecular names to uniform names. Default = {'N2h': 'N2', 'N2v': 'N2', 'NNHh': 'NNH', 'NNHv': 'NNH', 'NH': 'NH', 'NH3': 'NH3', 'H': 'H'}
        - deal_file (bool): whether to process structure files synchronously. Default = False
        - error_path (path): path to store site change and anomaly slab. if None, delete them directly. Default = None
        - dataframe (pd.DataFrame): dataframe with index as slab file name. if not None, process dataframe according to its index. Default = None
        - detect_anomaly (bool): whether to detect anomaly. Default = True
        - disable_tqdm (bool): Default = True
        - jupyter_tqdm (bool): Default = True
    Returns:
        - dfa: a DataFrame with anomaly slab if detect_anomaly
        - error_list: a list of site change slab
        - isomo_list: a list of slabs that confirm as an isomorphism
        - dfs: a DataFrame with suggested initial slab 
        - empty_list: list of slabs that site changed after optimization
    '''
    # detect anomaly
    if detect_anomaly:
        print('Start detect anomaly')
        dfa, el = detect_anoma(
            path_ori,
            path_opt,
            adsb_symbol,
            'optima',
            cutoff_multiplier,
            disable_tqdm,
            jupyter_tqdm
        )
        count_anomaly = dfa.shape[0]
        print(f'Find {count_anomaly} anomaly slabs.')
        if deal_file:
            if error_path is not None:
                for adss in dfa.index:
                    move(os.path.join(path_opt, adss), os.path.join(error_path, adss))
                    move(os.path.join(path_ori, adss), os.path.join(error_path, adss.split('.')[0] + '_initial.' + adss.split('.')[1]))
                print('Move anomaly slabs to error path')
            else:
                for adss in dfa.index:
                    os.remove(os.path.join(path_ori, adss))
                    os.remove(os.path.join(path_opt, adss))
                print('Remove anomaly slabs')
    # check site cnosistence same configuration
    print('Check site consistent for same slab')
    error_list, empty_list, isomo_list = site_consistent_same_slab(
        path_ori,
        path_opt,
        para_infer_site_origin,
        para_infer_site_optima,
        isomorphism_ori,
        isomorphism_opt,
        disable_tqdm,
        jupyter_tqdm
    )
    count_error = len(error_list)
    count_empty = len(empty_list)
    count_isomo = len(isomo_list)
    print(f'Consider {count_isomo} slabs as isomorphism')
    print(f'Find {count_error} slabs with change site')
    print(f'{count_empty} slabs have not initial slab')
    # find origin
    print('Suggest initial slab for site change slab')
    suggest_origin = find_origin(
        error_list,
        path_all_ori,
        path_opt,
        para_infer_site_origin,
        para_infer_site_optima,
        uniform_adsb,
        disable_tqdm,
        jupyter_tqdm
    )
    initial_list = []
    suggest_list = []
    alexist_list = []
    count_alexist = 0
    for adss in suggest_origin.keys():
        if suggest_origin[adss] is not None:
            initial_list.append(adss)
            suggest_list.append(suggest_origin[adss])
            if suggest_list[-1] in suggest_list[:-1]:
                alexist_list.append(True)
                count_alexist += 1
            elif os.path.exists(os.path.join(path_opt, suggest_origin[adss])):
                alexist_list.append(True)
                count_alexist += 1
            else:
                error_list.pop(error_list.index(adss))
                alexist_list.append(False)
    count_suggest = len(suggest_list)
    print(f'Find {count_suggest} suggest slab')
    print(f'{count_alexist} recommendation slabs already exist')
    # deal file
    if deal_file:
        if error_path is not None:
            for adss in error_list:
                move(os.path.join(path_opt, adss), os.path.join(error_path, adss))
                move(os.path.join(path_ori, adss), os.path.join(error_path, adss.split('.')[0] + '_initial.' + adss.split('.')[1]))
            print('Move site change slabs to error path')
        else:
            for adss in error_list:
                os.remove(os.path.join(path_opt, adss))
                os.remove(os.path.join(path_ori, adss))
            print('Remove site change slabs')
        for i, adss in enumerate(initial_list):
            if not alexist_list[i]:
                os.rename(os.path.join(path_opt, adss), os.path.join(path_opt, suggest_list[i]))
                os.remove(os.path.join(path_ori, adss))
                copy(os.path.join(path_all_ori, suggest_list[i]), os.path.join(path_ori, suggest_list[i]))
        print(f'Replace {count_suggest - count_alexist} slabs with recommendation slab')
    dfs = pd.DataFrame.from_dict({'initial adss': initial_list, 'suggest adss': suggest_list, 'already exist': alexist_list})
    # deal dataframe
    if dataframe is not None:
        dataframe.drop(list(dfa.index), inplace=True) if detect_anomaly else None
        dataframe.drop(error_list, inplace=True)
        initial_list_dropexist = [initial_list[ii] for ii in range(count_suggest) if not alexist_list[ii]]
        suggest_list_dropexist = [suggest_list[ii] for ii in range(count_suggest) if not alexist_list[ii]]
        dataframe.rename(dict(zip(initial_list_dropexist, suggest_list_dropexist)), inplace=True)
        print('Complete the deletion and correction of the dataframe')
        
    if detect_anomaly:
        return dfa, error_list, isomo_list, dfs, empty_list
    else:
        return error_list, isomo_list, dfs, empty_list

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
    origin_file_path,
    optima_file_path,
    para_infer_site_origin={},
    para_infer_site_optima=[{}],
    isomorphism_ori=[],
    isomorphism_opt=[],
    disable_tqdm=True,
    jupyter_tqdm=True
):
    '''
    Function to check whether adsorption sites of a same slab are consistent after optimization.

    Parameters:
        - origin_file_path (path): file paths to origin slabs.
        - optima_file_path (path): file paths to relaxed slabs.
        - para_infer_site_origin (dict): a dict of key value pairs of parameter on infer_adsorption_site for origin slab. Default = {}
        - para_infer_site_optima (list of dicts): a list of dicts of key value pairs of parameter on infer_adsorption_site for optimization slab.
          The program will use different parameters in turn to determine whether the site is changed. Default = [{}]
        - isomorphism_ori (list of dict): dicts of coordination. If an original slab has those coordination, then its optimized structure may have an 
          isomorphism. This corresponds to the coordination in isomorphism_opt.
        - isomorphism_opt (list of dict): dicts of coordination. Suppose a slab has this coordination, and the site is the same, and the connection
          relationship contains the original. In that case, this slab will also be considered the same configuration to the origin slab. Default = None.
        - disable_tqdm (bool). Default = False
        - jupyter_tqdm (bool). Default = True
    Return:
        - error_list (list): list of slabs that site changed after optimization.
        - empty_list (list): list of slabs that can not find a corresponding origin slab structure. 
        - isomo_list (list): list of slabs that confirm as an isomorphism.
    '''
    # init
    error_list = []
    empty_list = []
    isomo_list = []
    tqdm = tqdm_notebook if jupyter_tqdm else tqdm_common
    # loop for each optima adss
    opt_adss = get_file_or_subdirection(optima_file_path)
    with tqdm(opt_adss, disable=disable_tqdm) as pbar:
        for adss in pbar:
            # check origin file exist
            if not os.path.exists(os.path.join(origin_file_path, adss)):
                empty_list.append(adss)
                continue
            # get origin site info
            structure = Structure.from_file(os.path.join(origin_file_path, adss))
            osite, ocoor, ocnet = infer_adsorption_site(structure, **para_infer_site_origin)
            # get optima site info and check consistent
            structure = Structure.from_file(os.path.join(optima_file_path, adss))
            for para in para_infer_site_optima:
                psite, pcoor, pcnet = infer_adsorption_site(structure, **para)
                if psite == osite and ocoor == pcoor and  set([tuple(b) for b in np.array(ocnet).transpose()]) == set([tuple(b) for b in np.array(pcnet).transpose()]):
                    error = False
                    break
                elif ocoor in isomorphism_ori:
                    has_isomorphism = False
                    for ic, icoor in enumerate(isomorphism_ori):
                        if ocoor == icoor and psite == osite and pcoor == isomorphism_opt[ic] and set([tuple(b) for b in np.array(ocnet).transpose()]).issubset(set([tuple(b) for b in np.array(pcnet).transpose()])):
                            has_isomorphism = True
                            isomo_list.append(adss)
                            break
                    if has_isomorphism:
                        error = False
                        break
                else:
                    error = True
            if error:
                error_list.append(adss)
                
    return error_list, empty_list, isomo_list


def find_origin(
    error_list,
    path_origin_slab,
    path_optima_slab,
    para_infer_site_origin={},
    para_infer_site_optima=[{}],
    uniform_adsb={'N2h': 'N2', 'N2v': 'N2', 'NNHh': 'NNH', 'NNHv': 'NNH', 'NH': 'NH', 'NH3': 'NH3', 'H': 'H'},
    disable_tqdm=True,
    jupyter_tqdm=True,
):
    '''
    A function used to find a possible initial slab corresponding to a slab with a changing site after optimization.

    Parameters:
        - error_list (list): file names of site changing slab structures.
        - path_origin_slab (path): path to all constructed initial slab structures.
        - path_optima_slab (path): path to relaxed slab structures.
        - para_infer_site_origin (dict): a dict of key value pairs of parameter on infer_adsorption_site for origin slab. Default = {}
        - para_infer_site_optima (list of dicts): a list of dicts of key value pairs of parameter on infer_adsorption_site for optimization slab. Default = [{}]
        - uniform_adsb (dict): convert molecular names to uniform names. 
          Default = {'N2h': 'N2', 'N2v': 'N2', 'NNHh': 'NNH', 'NNHv': 'NNH', 'NH': 'NH', 'NH3': 'NH3', 'H': 'H'}
        - disable_tqdm (bool). Default = False
        - jupyter_tqdm (bool). Default = True
    Return:
        - suggest_origin (dict): a dict for a suggest initial slab for each site changing slab
    '''
    tqdm = tqdm_notebook if jupyter_tqdm else tqdm_common
    # init empty data list
    suggest_origin  = {}
    # extract origin slab and adsb info
    origin_strucs = get_file_or_subdirection(path_origin_slab)
    origin_slabs  = np.array([adss.split('_')[0] for adss in origin_strucs])
    origin_adsbs  = np.array([uniform_adsb[adss.split('_')[1]] for adss in origin_strucs])
    # loop for each error adss
    with tqdm(error_list, disable=disable_tqdm) as pbar:
        for adss in pbar:
            eslab = adss.split('_')[0]
            eadsb = uniform_adsb[adss.split('_')[1]]
            # collect origin adss site
            origin_sites = []
            index = np.where(origin_slabs == eslab)[0]
            index = index[np.where(origin_adsbs[index] == eadsb)[0]]
            alternative = [origin_strucs[i] for i in index]
            for oadss in alternative:
                structure = Structure.from_file(os.path.join(path_origin_slab, oadss))
                osite, ocoor, ocnet = infer_adsorption_site(structure, **para_infer_site_origin)
                origin_sites.append((osite, ocoor, ocnet))
            # get possible site for eadss
            eadss_sites = []
            eadss = Structure.from_file(os.path.join(path_optima_slab, adss))
            for para in para_infer_site_optima:
                esite, ecoor, ecnet = infer_adsorption_site(eadss, **para)
                eadss_sites.append((esite, ecoor, ecnet))
            # find suggest origin
            find = False
            for esc in eadss_sites:
                for oi, osc in enumerate(origin_sites):
                    if esc[0] == osc[0] and esc[1] == osc[1] and set([tuple(b) for b in np.array(esc[2]).transpose()]) == set([tuple(b) for b in np.array(osc[2]).transpose()]):
                        find = True
                        suggest_adss = alternative[oi]
                        break
            # store result
            if find:
                suggest_origin[adss] = suggest_adss
            else:
                suggest_origin[adss] = None
                
        return suggest_origin


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


def detect_anoma(
    path_initial,
    path_optima,
    adsb_symbol=['N', 'H'],
    get_tag_by='initia',
    cutoff_multiplier=1.5,
    disable_tqdm=True,
    jupyter_tqdm=True,
):
    '''
    Function to detect anomalies based on initial and final stucture of a relaxation.

    Parameters:
        - path_initial (path): path to the initial slabs.
        - path_optima (path): path to the optimized slabs.
        - adsb_symbol (list): element symbols for adsorbate. Default = ['N', 'H']
          The program distinguishes molecules, surfaces, and blocks based on adsorbed elements and whether they are fixed.
        - get_tag_by (str): use 'initia' or 'optima' slab to distinguish atomic categories. Default = 'initia'.
        - cutoff_multiplier (float): the radius multiple used for the criterion. Default = 1.5.
        - disable_tqdm (bool)
        - jupyter_tqdm (bool)
    Returns:
        - A DataFrame that stores the abnormal structure and its corresponding cause
        - A list that stores structure names that do not match the corresponding initial structure
    '''
    # init
    anoma_list = []
    adsb_disso = []
    adsb_desor = []
    surf_chang = []
    adsb_inter = []
    empty_list = []
    tqdm = tqdm_notebook if jupyter_tqdm else tqdm_common
    # loop for each optima adss
    optima_adsses = get_file_or_subdirection(path_optima)
    with tqdm(optima_adsses, disable=disable_tqdm) as pbar:
        for adss in pbar:
            if not os.path.exists(os.path.join(path_initial, adss)):
                empty_list.append(adss)
                continue
            slab_init = Structure.from_file(os.path.join(path_initial, adss))
            slab_opti = Structure.from_file(os.path.join(path_optima, adss))
            detector = DetectTrajAnomaly(
                AseAtomsAdaptor.get_atoms(slab_init), 
                AseAtomsAdaptor.get_atoms(slab_opti), 
                get_atom_tag(slab_init , adsb_symbol) if get_tag_by == 'initia' else get_atom_tag(slab_opti, adsb_symbol),
                None, cutoff_multiplier, cutoff_multiplier
            )
            anoma = [
                detector.is_adsorbate_dissociated(),
                detector.is_adsorbate_desorbed(),
                detector.has_surface_changed(),
                detector.is_adsorbate_intercalated()
            ]
            if any(anoma):
                anoma_list.append(adss)
                adsb_disso.append(anoma[0])
                adsb_desor.append(anoma[1])
                surf_chang.append(anoma[2])
                adsb_inter.append(anoma[3])
    df = pd.DataFrame.from_dict({
        'adsorbate_dissociated': adsb_disso,
        'adsorbate_desorbed': adsb_desor,
        'surface_changed': surf_chang,
        'adsorbate_intercalated': adsb_inter
    })
    df.index = anoma_list
    
    return df, empty_list


def get_atom_tag(slab, adsb_ele=['N', 'H']):
    atom_number = len(slab)
    adsorbate_index = []
    for atom_index in range(atom_number):
        if slab.species[atom_index].symbol in adsb_ele:
            adsorbate_index.append(atom_index)
    ase_slab = AseAtomsAdaptor.get_atoms(slab)
    constraint_index = ase_slab.constraints[0].index
    tags = [0 if atom_index in constraint_index else 1 for atom_index in range(atom_number)]
    tags = [2 if atom_index in adsorbate_index else tags[atom_index] for atom_index in range(atom_number)]
    
    return tags
    

class DetectTrajAnomaly:
    def __init__(
        self,
        init_atoms,
        final_atoms,
        atoms_tag,
        final_slab_atoms=None,
        surface_change_cutoff_multiplier=1.5,
        desorption_cutoff_multiplier=1.5,
    ):
        """
        Flag anomalies based on initial and final stucture of a relaxation. Copy from fairchem.

        Args:
            init_atoms (ase.Atoms): the adslab in its initial state
            final_atoms (ase.Atoms): the adslab in its final state
            atoms_tag (list): the atom tags; 0=bulk, 1=surface, 2=adsorbate
            final_slab_atoms (ase.Atoms, optional): the relaxed slab if unspecified this defaults
            to using the initial adslab instead.
            surface_change_cutoff_multiplier (float, optional): cushion for small atom movements
                when assessing atom connectivity for reconstruction
            desorption_cutoff_multiplier (float, optional): cushion for physisorbed systems to not
                be discarded. Applied to the covalent radii.
        """
        self.init_atoms = init_atoms
        self.final_atoms = final_atoms
        self.final_slab_atoms = final_slab_atoms
        self.atoms_tag = atoms_tag
        self.surface_change_cutoff_multiplier = surface_change_cutoff_multiplier
        self.desorption_cutoff_multiplier = desorption_cutoff_multiplier

        if self.final_slab_atoms is None:
            slab_idxs = [idx for idx, tag in enumerate(self.atoms_tag) if tag != 2]
            self.final_slab_atoms = self.init_atoms[slab_idxs]

    def is_adsorbate_dissociated(self):
        """
        Tests if the initial adsorbate connectivity is maintained.

        Returns:
            (bool): True if the connectivity was not maintained, otherwise False
        """
        adsorbate_idx = [idx for idx, tag in enumerate(self.atoms_tag) if tag == 2]
        return not (
            np.array_equal(
                self._get_connectivity(self.init_atoms[adsorbate_idx]),
                self._get_connectivity(self.final_atoms[adsorbate_idx]),
            )
        )

    def has_surface_changed(self):
        """
        Tests bond breaking / forming events within a tolerance on the surface so
        that systems with significant adsorbate induces surface changes may be discarded
        since the reference to the relaxed slab may no longer be valid.

        Returns:
            (bool): True if the surface is reconstructed, otherwise False
        """
        surf_idx = [idx for idx, tag in enumerate(self.atoms_tag) if tag != 2]

        adslab_connectivity = self._get_connectivity(self.final_atoms[surf_idx])
        slab_connectivity_w_cushion = self._get_connectivity(
            self.final_slab_atoms, self.surface_change_cutoff_multiplier
        )
        slab_test = 1 in adslab_connectivity - slab_connectivity_w_cushion

        adslab_connectivity_w_cushion = self._get_connectivity(
            self.final_atoms[surf_idx], self.surface_change_cutoff_multiplier
        )
        slab_connectivity = self._get_connectivity(self.final_slab_atoms)
        adslab_test = 1 in slab_connectivity - adslab_connectivity_w_cushion

        return any([slab_test, adslab_test])

    def is_adsorbate_desorbed(self):
        """
        If the adsorbate binding atoms have no connection with slab atoms,
        consider it desorbed.

        Returns:
            (bool): True if there is desorption, otherwise False
        """
        adsorbate_atoms_idx = [
            idx for idx, tag in enumerate(self.atoms_tag) if tag == 2
        ]
        surface_atoms_idx = [idx for idx, tag in enumerate(self.atoms_tag) if tag != 2]
        final_connectivity = self._get_connectivity(
            self.final_atoms, self.desorption_cutoff_multiplier
        )

        for idx in adsorbate_atoms_idx:
            if sum(final_connectivity[idx][surface_atoms_idx]) >= 1:
                return False
        return True

    def _get_connectivity(self, atoms, cutoff_multiplier=1.0):
        """
        Generate the connectivity of an atoms obj.

        Args:
            atoms (ase.Atoms): object which will have its connectivity considered
            cutoff_multiplier (float, optional): cushion for small atom movements when assessing
                atom connectivity

        Returns:
            (np.ndarray): The connectivity matrix of the atoms object.
        """
        cutoff = natural_cutoffs(atoms, mult=cutoff_multiplier)
        ase_neighbor_list = neighborlist.NeighborList(
            cutoff, self_interaction=False, bothways=True
        )
        ase_neighbor_list.update(atoms)
        matrix = neighborlist.get_connectivity_matrix(ase_neighbor_list.nl).toarray()
        return matrix

    def is_adsorbate_intercalated(self):
        """
        Ensure the adsorbate isn't interacting with an atom that is not allowed to relax.

        Returns:
            (bool): True if any adsorbate atom neighbors a frozen atom, otherwise False
        """
        adsorbate_atoms_idx = [
            idx for idx, tag in enumerate(self.atoms_tag) if tag == 2
        ]
        frozen_atoms_idx = [idx for idx, tag in enumerate(self.atoms_tag) if tag == 0]
        final_connectivity = self._get_connectivity(
            self.final_atoms,
        )

        for idx in adsorbate_atoms_idx:
            if sum(final_connectivity[idx][frozen_atoms_idx]) >= 1:
                return True
        return False