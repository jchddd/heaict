import os
import itertools
import pandas               as pd
import numpy                as np
from tqdm                   import tqdm as tqdm_common
from tqdm.notebook          import tqdm as tqdm_notebook
import matplotlib.pyplot    as plt
from ase.db                 import connect
from pymatgen.core          import Structure, Element
from pymatgen.io.ase        import AseAtomsAdaptor
import torch
from torch_geometric.data   import Data
from torch_geometric.utils  import dense_to_sparse
from heaict.data.utility import get_file_or_subdirection, find_indices, get_element_symbol, read_mag_outcar, check_existence
from heaict.data.site    import infer_adsorption_site


def load_graph_from_database(
    database,
    targets=['target'],
    node_targets=[],
    edge_targets=[],
    save=False,
    pt_file=r'./graphs.pt',
    disable_tqdm=True,
    jupyter_tqdm=True,
    para_grab_ele_feature={},
    para_construct_graph={},
    para_add_feature={},
):
    '''
    Function to construct graphs from ase database, add features and targets.
    
    Parameters:
        - database (path str): path to the ase database
        - targets (list or None): list of keys for target values in database. Default = ['target']
        - node_targets (list): list of keys for node target arrays in database. Default = []
        - edge_targets (list): list of keys for edge target arrays in database. Default = []
        - save (bool): save graphs to pt file. Default = False
        - pt_file (file str): name of the saved pt file. Default = r'./graphs.pt'
        - disable_tqdm (bool). Default = True
        - jupyter_tqdm (bool). Default = True
        - para_grab_ele_feature (dict): parameters for grab_element_feature. Default = {}
          Template: {'features': 'All', 'normalization': True, 'use_atomber_onehot': False}. If 'elements' is not passed, elements will be graded from the database automatically.
        - para_construct_graph (dict): parameters on construct_graph and infer_adsorption_site. Default = {}
          Template: {'cutoff_substrate': 3, 'cutoff_adsorbate': 1.3, 'cutoff_bond': 2.7, 'adsb': ['N', 'H'], 'forbidden_bonds': [['H', 'H']]}
          And other key-value pairs in infer_adsorption_site except 'slab', 'adsb', and 'judge'
        - para_add_feature (dict): parameters for add_features except 'targets', 'graph', and 'df_feature'. Default = {}
          Template: {'coordinate': [9, 12], 'distance_feature': True, 'distance_classes': None}
    Return:
        - The list of graphs
    '''
    graphs = []
    # connect database
    database = connect(database)
    # grab element features
    if 'elements' not in para_grab_ele_feature.keys(): para_grab_ele_feature.update({'elements': get_element_symbol(database)})
    df_feature = grab_element_feature(**para_grab_ele_feature)
    # read and construct graph
    tqdm = tqdm_notebook if jupyter_tqdm else tqdm_common
    with tqdm(range(len(database)), disable=disable_tqdm) as pbar:
        for i in pbar:
            row   = database[i + 1]
            slab  = AseAtomsAdaptor.get_structure(row.toatoms())
            graph = construct_graph(slab, **para_construct_graph)
            grapy = [float(getattr(row, target)) for target in targets]
            nodey = np.concatenate([row.data[target] for target in node_targets], axis=-1) if len(node_targets) > 0 else None
            edgey = np.concatenate([row.data[target] for target in edge_targets], axis=-1) if len(edge_targets) > 0 else None
            graph = add_features(graph, df_feature, grapy, nodey, edgey, **para_add_feature)
            # add to list
            graphs.append(graph)
    # save graphs to pt file
    if save:
        torch.save(graphs, pt_file)
    
    return graphs


def build_database(
    slabs_direction,
    df_target=None,
    targets=['target'],
    targets_array=[],
    database=r'./slab.db',
    disable_tqdm=True,
    jupyter_tqdm=True,
    adsorbate_elements=['N', 'H'],
    para_n_neib_slab={},
    para_infer_ads_site={}
):
    '''
    Function to build ase database from structures
    
    Parameters:
        - slabs_direction (path str): path to where store all input POSCARs
        - df_target (pd.DataFrame): DataFrame that stores target values with slab file names as the index, None is not written targets. Default = None
        - targets (list): column names for targets in csv file. Default = ['target']
          Each target value is assigned to a key-value pair in the database row. Some bad keys like energy can not be used.
        - target_array (list) column names for array targets in csv file. Default = []
        - database (str, file): path to the created database file. Default = r'./slab.db'
        - disable_tqdm (bool). Default = True
        - jupyter_tqdm (bool). Default = True
        - adsorbate_elements (list): list of adsorbate elements for n_neighbor_slab construction. Default = ['N', 'H']
        - para_n_neib_slab (dict or None): parameters on get_n_neighbor_slab, used to convert slab to n_neighbor_slab. None is store origin structures. Default = {}
          Template: {'number_neighbors': 2, 'max_neighbor_number': 12, 'cutoff': 3}
        - para_infer_ads_site (dict): parameters on infer_adsorption_site, used to determine the number of atoms on adsorption site for n_neighbor_slab construction. Default = {}
          Template: {'adsb_excl': [], 'method': 'cutoff', 'judge': 2.7, defin_dist: 'dist'}
    '''
    # load target file
    # if target_csv is not None:
    #     df_target = pd.read_csv(target_csv, index_col=0)
    # check if the database exists, and connect it 
    if os.path.exists(database):
        print(f'Warning !!! {database} exists')
    database = connect(database)
    # loop for each slab
    slabs = get_file_or_subdirection(slabs_direction)
    tqdm = tqdm_notebook if jupyter_tqdm else tqdm_common
    with tqdm(slabs, disable=disable_tqdm) as pbar:
        for slab in pbar:
            # get target values from target csv
            target_values = {'struc_name': slab.split('.')[0]}
            if df_target is not None:
                for target in targets:
                    target_values[target] = float(df_target.loc[slab, target])
            # get array target from target csv
            array_data = {}
            if df_target is not None and len(targets_array) > 0:
                for target_array in targets_array:
                    array_data[target_array] = df_target.loc[slab, target_array]
            # get slab structure
            slab = Structure.from_file(os.path.join(slabs_direction, slab))
            if 'selective_dynamics' in slab.site_properties.keys():
                slab.remove_site_property('selective_dynamics')
            # get n neighbor slab
            if para_n_neib_slab is not None:
                site, coor, cnet = infer_adsorption_site(slab, adsorbate_elements, **para_infer_ads_site)
                slab = get_n_neighbor_slab(slab, len(site), adsorbate_elements, **para_n_neib_slab)
            # write slab and target to database
            slab = AseAtomsAdaptor.get_atoms(slab)
            database.write(slab, **target_values, data=array_data)


def get_n_neighbor_slab(
    slab, 
    number_site_atoms, 
    adsorbate_elements=['H', 'N'], 
    number_neighbors=3, 
    max_neighbor_number=66,
    cutoff=3,
):
    '''
    Functin to find n neighbor for adsorbate and construct the corresponding slab.
    The first neighbor refers to the adsorption site itself.
    
    Parameters:
        - slab (pymatgen.core.Structure).
        - number_site_atoms (int): number of adsorption site atoms. Default = 3
        - adsorbate_elements (list). Default = ['H', 'N']
        - number_neighbors (int): the n of n neighbors. Default = 3
        - max_neighbor_number (int): max neighbor number for substrate slab atoms. Default = 66
        - cutoff (float): The neighbor cutoff radius of substrate atoms. Default = 3
    Return:
        - The n neighbor slab (pymatgen.core.Structure).
    '''
    number_layers = number_neighbors
    scale_to_substrate = 3
    # create scaled substrate
    if 'selective_dynamics' in slab.site_properties.keys():
        slab.remove_site_property('selective_dynamics')
    substrate = slab.copy()
    substrate.remove_species(adsorbate_elements)
    substrate.make_supercell([scale_to_substrate] * 2 + [1])
    # add adsorbate to substrate
    adsorbate_indices = find_indices(slab.species, [Element(e) for e in adsorbate_elements])
    adsorbate_sites = [slab[i] for i in adsorbate_indices]
    vector_to_center_image = (slab.lattice.matrix * np.array([1,1,0])).sum(axis=0)
    adsorbate_coords = _coord_to_one_image(slab, adsorbate_indices)
    for i, site in enumerate(adsorbate_sites):
        substrate.append(site.specie, adsorbate_coords[i] + vector_to_center_image, True)
    # find n neighbors 
    adsorbate_indices = find_indices(substrate.species, [Element(e) for e in adsorbate_elements])
    neighbor_slab_indices = adsorbate_indices
    source_atom_indices = adsorbate_indices
    for n in range(number_neighbors):
        if n == 0: max_neighbor_number_for_n = number_site_atoms
        else: max_neighbor_number_for_n = max_neighbor_number
        n_neighbors = _find_neighbors(substrate, source_atom_indices, max_neighbor_number_for_n, cutoff=cutoff)
        source_atom_indices = n_neighbors
        neighbor_slab_addition = n_neighbors[np.isin(n_neighbors, neighbor_slab_indices, invert=True)]
        neighbor_slab_indices = np.concatenate((neighbor_slab_indices, neighbor_slab_addition))
    # construct neighbor slab
    substrate_indices = np.arange(0, len(substrate))
    remove_indices = substrate_indices[np.isin(substrate_indices, neighbor_slab_indices, invert=True)]
    substrate.remove_sites(remove_indices)
    
    return substrate


def _find_neighbors(
    slab, 
    source_atom_indices, 
    max_neighbor_number=66, 
    cutoff=3
):
    '''
    Function to find neighbors for a series of atoms.
    
    Parameters:
        - slab (pymatgen.core.Structure).
        - source_atom_indices (array like).
        - max_neighbor_number (int). Default = 66
        - cutoff (float). Default = 3
    Return:
        - Indices for neighbors (list).
    '''
    number_source_atoms = len(source_atom_indices)
    number_slab_atoms = len(slab)
    min_distance_to_source = np.min(np.array([slab.distance_matrix[i] for i in source_atom_indices]), axis=0)
    mask_cutoff = min_distance_to_source > cutoff
    min_distance_to_source_masked = np.ma.array(min_distance_to_source, mask=mask_cutoff)
    sorted_indices = np.argsort(min_distance_to_source_masked)
    number_mask_atoms = np.count_nonzero(mask_cutoff)
    neighbors_indices = sorted_indices[: min(max_neighbor_number + number_source_atoms, number_slab_atoms - number_mask_atoms)]
    mask_source = np.isin(neighbors_indices, source_atom_indices, invert=True)
    neighbors_indices = neighbors_indices[mask_source]
    return neighbors_indices


def _coord_to_one_image(slab, indices):
    '''
    move coordinate of adsorbate atoms into one image

    Parameters:
        - slab (pymatgen.Structure)
        - indices (int array-like): indices of adsorbate atoms
    Return:
        - New Cartesian coordinate of adsorbate atoms 
    '''
    slab = slab.copy()
    center = np.sum(slab.lattice.matrix[:2][:, :2] / 2, axis=0)
    adsb_c2d = slab.cart_coords[indices][:, :2]
    dist_to_center = np.linalg.norm(adsb_c2d - center, axis=1)
    ref_point = indices[np.argsort(dist_to_center)[0]]

    image_move = np.array([
        [-1., -1.],
        [-1.,  0.],
        [-1.,  1.],
        [ 0., -1.],
        [ 0.,  0.],
        [ 0.,  1.],
        [ 1., -1.],
        [ 1.,  0.],
        [ 1.,  1.]
    ])
    fc_ref = slab.frac_coords[ref_point]
    for ai in indices:
        if   ai == ref_point:
            continue
        else:
            fc_ai = slab.frac_coords[ai]
            fc_ai_image = fc_ai[: 2] + image_move
            fc_ai_neare = fc_ai_image[np.argsort(np.linalg.norm(fc_ai_image - fc_ref[:2], axis=1))[0]]
            slab.replace(ai, slab[ai].species, np.concatenate([fc_ai_neare, fc_ai[2: 3]]))
            
    return slab.cart_coords[indices]


def construct_graph(
    slab,
    cutoff_substrate=3,
    cutoff_adsorbate=1.3,
    cutoff_bond=2.7,
    adsb=['N', 'H'],
    forbidden_bonds=[['H', 'H']],
    add_distance=False,
    all_connect=False,
    print_info=False,
    para_infer_ads_site={}
):
    '''
    Function to construct graph from a slab structure. The structure will be divided into substrate and adsorbate.
    Then processes connection relationship between them. Finally connection relationships among them are processed separately.
    
    Parameters:
        - slab (pymatgen.core.Structure).
        - cutoff_substrate (float): cutoff radius for substrate. Default = 3
        - cutoff_adsorbate (float): cutoff radius for adsorbate. Default = 1.3
        - cutoff_bond (float): cutoff radius for bond between substrate and adsorbate. Default = 2.7
        - adsb (list): elements on adsorbate. Default = ['N', 'H']
        - forbidden_bonds ([n,2] list): forbidden bonds on adsorbate. Default = [['H', 'H']]
        - add_distance (bool): whether to add interatomic_distance to edge feature. Default = False
        - all_connect (bool): connect all sites atoms to adsorbate bonding atoms and add edge_type attribute to graph. Default = False
        - print_info (bool): whether to print bonding information or not. Default = False
        - para_infer_ads_site (dict): parameters for infer_adsorption_site except judge and adsb, better to use 'dist' radius.
    Return:
        - The graph (torch_geometric.data.Data).
        - following attribute will be added to graph:
            >> edge_index; atomic_number; atomic_type; interatomic_distance: added to graph is add_distance
            >> edge_type: used to distinguish the bonds between metals and molecules from other bonds, only added when using the all_comnect command
            >> bond_pair: Node index of all combinations of adsorption sites and molecule atoms bonded to the surface
            >> num_bond: The length (number) of bond_pair >> bonded_pair: bonding situation for each bond pair
    Cautious:
        - Note is not suitable for a periodic structure
    '''
    # find adsorbate and adsorption site
    site, coor, cnet = infer_adsorption_site(slab, judge=cutoff_bond, adsb=adsb, **para_infer_ads_site)
    number_sites = len(slab)
    slab_indices = np.arange(number_sites)
    adsorbate_indices = find_indices(slab.species, [Element(e) for e in adsb])
    adsorption_site_indices = np.array(list(set(cnet[1])))
    substrate_indices = slab_indices[np.isin(slab_indices, adsorbate_indices, invert=True)]
    # get mask_substrate and mask bond between substrate and adsorbate
    distance_matrix = slab.distance_matrix
    mask_substrate = distance_matrix > cutoff_substrate
    row_grid, col_grid = np.meshgrid(substrate_indices, adsorbate_indices, indexing='ij')
    mask_substrate[row_grid.ravel(), col_grid.ravel()] = True
    # get mask_adsorbate and mask bond between adsorbate and substrate 
    mask_adsorbate = distance_matrix > cutoff_adsorbate
    mask_adsorbate[col_grid.ravel(), row_grid.ravel()] = True
    # deal forbidden bond
    for forbidden_bond in forbidden_bonds:
        forbidden_indices0 = find_indices(slab.species, [Element[forbidden_bond[0]]])
        forbidden_indices1 = find_indices(slab.species, [Element[forbidden_bond[1]]])
        mask_equal = np.equal(forbidden_indices0, forbidden_indices1)
        forbidden_indices0_filtered = forbidden_indices0[~mask_equal]
        forbidden_indices1_filtered = forbidden_indices1[~mask_equal]
        if len(forbidden_indices0_filtered) == len(forbidden_indices1_filtered) and len(forbidden_indices0_filtered) != 0:
            mask_adsorbate[forbidden_indices0, forbidden_indices1] = True
            mask_adsorbate[forbidden_indices1, forbidden_indices0] = True
    # update mask_substrate by mask_adsorbate
    mask_substrate[adsorbate_indices] = mask_adsorbate[adsorbate_indices]
    # add bond between substrate and adsorbate
    if all_connect and len(site) == 3:
        combinations = list(set(itertools.product(cnet[0], cnet[1])))
        cnetac = [[item[0] for item in combinations], [item[1] for item in combinations]]
        combinations = np.array(combinations)
    else:
        cnetac = cnet
        combinations = np.array(cnetac).transpose()
    mask_substrate[cnetac[0] + cnetac[1], cnetac[1] + cnetac[0]] = False
    combinations[np.lexsort((combinations[:, 1], combinations[:, 0]))]
    # update distance matrix
    distance_matrix_masked = np.ma.array(distance_matrix, mask=mask_substrate)
    distance_matrix_masked = np.argsort(np.argsort(distance_matrix_masked))
    distance_matrix_masked = np.nan_to_num(np.where(mask_substrate, np.nan, distance_matrix_masked))
    distance_matrix_masked = np.where(distance_matrix_masked == 0, distance_matrix_masked, distance_matrix)
    # print bond info
    if print_info:
        for site in range(number_sites):
            print('site %3d - %-2s' % (site, slab[site].specie.symbol))
            print('|', end='')
            for sitej, distance in enumerate(distance_matrix_masked[site]):
                if distance > 0:
                    print('%3d-%-2s-%3.2f|' % (sitej, slab[sitej].specie.symbol, distance), end='')
            print()
    # generate graph related info
    distance_matrix_masked = torch.Tensor(distance_matrix_masked)
    edge_info = dense_to_sparse(distance_matrix_masked)
    edge_index = edge_info[0]
    edge_weight = edge_info[1]
    atomic_type = torch.zeros(number_sites)
    atomic_type[adsorbate_indices] = 2
    atomic_type[adsorption_site_indices] = 1
    if all_connect:
        all_pair = edge_index.t().unsqueeze(1)
        bond_pair = torch.Tensor(combinations).unsqueeze(0)
        matches = all_pair == bond_pair
        matches = torch.all(matches, dim=-1)
        edge_type = torch.any(matches, dim=-1)
        edge_type = edge_type.to(torch.int32)
    # construct graph data
    data = Data()
    data.edge_index = edge_index
    data.edge_type = edge_type if all_connect else None
    data.interatomic_distance = edge_weight if add_distance else None
    data.atomic_number = torch.Tensor(slab.atomic_numbers).long()
    data.atomic_type = atomic_type
    # bond info
    bond_pair = torch.Tensor(combinations).long()
    _, indices1 = torch.sort(bond_pair[:, 1])
    bond_pair = bond_pair[indices1]
    _, indices2 = torch.sort(bond_pair[:, 0])
    bond_pair = bond_pair[indices2]
    data.bond_pair = bond_pair
    data.num_bond = torch.Tensor([data.bond_pair.shape[0]]).long()
    data.bonded_pair = check_existence(data.bond_pair, torch.Tensor(cnet).t())

    return data


def add_features(
    graph, 
    df_feature,
    targets=None,
    nodey=None,
    edgey=None,
    coordinate=[9, 12],
    distance_feature=True,
    distance_classes=None,
    use_NRRBOP=False,
    bond_feature=False,
    add_node_distance=True
):
    '''
    Function to add features (node and edge feature, node cluster and target) to a graph
    
    Parameters:
        - graph (torch_geometric.Data).
        - df_feature (pd.DataFrame): dataframe that store node features.
        - targets (list): target values stored in a list. Default = None
        - nodey (array like): graph target values. Default = None
        - edgey (array like): edge target values. Default = None
        - coordinate (list or None): coordinate numbers for surface and bulk. Default = [9, 12]
          This adds geometric features that determine whether the atoms and edges are in a bulk, boundary, or surface. Set None to turn off.
        - distance_feature (bool): add graph distance related features to node and edge. Default = True
        - distance_classes (int or None): manual setting number classes of node distance, Recommend setting to n_neighbor + 1. Default = None
        - use_NRRBOP (bool): Use NRR related bond order and hybridization edge attribute. Default = False
        - bond_feature (bool): Use bond feature for edge between metal and molecular, used if all_connect when construct_graph. Default = False
        - add_node_distance(bool): Add node distance node_attr to graph. Default = True
    Return:
        - The graph with features
    '''
    # elemental features
    ## add node feature from df_feature
    node_features = []
    for atomic_number in graph.atomic_number:
        node_features.append(torch.Tensor(df_feature.loc[int(atomic_number), :].values))
    graph.x = torch.vstack(node_features)
    ## add bond_type to edge_attr (covalent, Ionic, Metal)
    bond_type = torch.zeros(graph.num_edges)
    for edge_index in range(graph.num_edges):
        bonding_pairs = graph.atomic_type[graph.edge_index[:, edge_index]]
        bond_type_judge = bonding_pairs[0] * bonding_pairs[1]
        if bond_type_judge > 3: # Intermolecular covalent bonds 2*2
            bond_type[edge_index] = 2
        elif bond_type_judge > 1: # Ionic bonds between molecules and metals 1*2
            bond_type[edge_index] = 1
        else: # Metal bonds between metals 0*0 0*1
            bond_type[edge_index] = 0
    graph.edge_attr = torch.nn.functional.one_hot(bond_type.long(), num_classes=3).float()
    # Geometric feature (bulk, surface, edge, adsorbate)
    if coordinate is not None:
        ## add node type 
        node_positioin = torch.zeros(graph.num_nodes)
        for i in range(graph.num_nodes):
            if   (graph.atomic_type[i] == 2).item(): # adsorbate
                node_positioin[i] = 0
            elif (graph.atomic_type[i] == 1).item(): # surface(site)
                node_positioin[i] = 1
            else:
                ligancy = torch.count_nonzero(graph.edge_index[0] == i).item()
                if   ligancy < coordinate[0]: # edge
                    node_positioin[i] = 2
                elif ligancy >= coordinate[0] and ligancy < coordinate[1]: # surface
                    node_positioin[i] = 1
                elif ligancy == coordinate[1]: #bulk
                    node_positioin[i] = 3
        graph.x = torch.cat([graph.x,  torch.nn.functional.one_hot(node_positioin.long(), num_classes=4).float()], dim=-1)
        ## add edge position
        edge_position = torch.zeros(graph.num_edges)
        for edge_index in range(graph.num_edges):
            node_pairs = node_positioin[graph.edge_index[:, edge_index]]
            position_judge = node_pairs[0] * node_pairs[1]
            if   position_judge == 0:
                edge_position[edge_index] = 0 # connect to adsorbate 
            elif position_judge == 1:
                edge_position[edge_index] = 1 # edge on surface 
            elif position_judge in [3, 9]:
                edge_position[edge_index] = 2 # edge on internal bulk
            elif position_judge == 4:
                edge_position[edge_index] = 3 # edge on edge
            elif position_judge in [2, 6]:
                edge_position[edge_index] = 4 # edge linking edge and surface, bulk
        graph.edge_attr = torch.cat([graph.edge_attr, torch.nn.functional.one_hot(edge_position.long(), num_classes=5).float()], dim=-1)
    # distance feature (distance to adsorbate)
    ## node distance
    if distance_feature or add_node_distance:
        distance = 0
        node_distance = torch.zeros(graph.num_nodes) - 1
        current_neighbor = torch.where(graph.atomic_type == 2)[0]
        node_distance[current_neighbor] = distance 
        while torch.count_nonzero(node_distance == -1).item() > 0:
            distance += 1
            next_neighbor = []
            for node in current_neighbor:
                edges_include_node = graph.edge_index[:, graph.edge_index[0] == node]
                next_neighbor.append(edges_include_node[1])
            next_neighbor = torch.unique(torch.cat(next_neighbor, dim=-1))
            next_neighbor = next_neighbor[node_distance[next_neighbor] == -1]
            node_distance[next_neighbor] = distance
            current_neighbor = next_neighbor
    if add_node_distance:
        graph.node_distance = node_distance.long()
    if distance_feature:
        dis_classes = distance_classes if distance_classes is not None else distance + 1
        graph.x = torch.cat([graph.x, torch.nn.functional.one_hot(node_distance.long(), num_classes=dis_classes).float()], dim=-1)
    ## edge distance
        edge_distance = torch.zeros(graph.num_edges)
        for edge_index in range(graph.num_edges):
            node_distances = node_distance[graph.edge_index[:, edge_index]]
            if   node_distances[0] == node_distances[1] and node_distances[0] != 0:
                distance = max(node_distances) + 1
            else:
                distance = max(node_distances)
            edge_distance[edge_index] = distance
        dis_classes = distance_classes + 1 if distance_classes is not None else max(edge_distance).long().item() + 1
        graph.edge_attr = torch.cat([graph.edge_attr, torch.nn.functional.one_hot(edge_distance.long(), num_classes=dis_classes).float()], dim=-1)
    # add node_cluster
    graph.node_cluster = graph.atomic_type.long()
    #graph.atomic_type = torch.nn.functional.one_hot(graph.atomic_type.long(), num_classes=3).float()
    # add bond order, hybridization
    if use_NRRBOP:
        num_H = torch.count_nonzero(graph.atomic_number == 1).item()
        num_N = torch.count_nonzero(graph.atomic_number == 7).item()
        num_B = torch.count_nonzero(bond_type == 1).item() / 2
        ## bond order
        if   num_N == 2 and num_H == 1: # NNH, NN double bond
            bond_order = 1
        elif num_N == 2 and num_H == 0: # N2, NN triple bond
            bond_order = 2
        bo = torch.zeros(graph.num_edges)
        i_to_zero = []
        for i, ei in enumerate(range(graph.num_edges)):
            ans = set([graph.atomic_number[graph.edge_index[0][ei]].item(), graph.atomic_number[graph.edge_index[1][ei]].item()])
            if   ans == {7}:
                bo[ei] = bond_order
            elif ans == {1, 7}:
                bo[ei] == 0 # N-H, single bond oreder
            else:
                i_to_zero.append(i)
        bo = torch.nn.functional.one_hot(bo.long(), num_classes=3).float()
        bo[i_to_zero, :] = 0
        ## hybridization
        if   num_B == 1: # vertice adsorption at top site
            hybridization = 0
        elif num_B != 1:
            hybridization = 1
        hb = torch.zeros(graph.num_edges)
        hb[torch.where(bond_type == 1)[0]] = hybridization
        hb = torch.nn.functional.one_hot(hb.long(), num_classes=2).float()
        hb[torch.where(bond_type != 1)[0], :] = 0
        graph.edge_attr = torch.cat([graph.edge_attr, bo, hb], dim=-1)
    if bond_feature:
        ma_pair_is_bond  = check_existence(graph.edge_index.t(), torch.cat([graph.bond_pair[graph.bonded_pair.squeeze(dim=-1)  == 1, :], graph.bond_pair[graph.bonded_pair.squeeze(dim=-1)  == 1, :][:, [1, 0]]])).squeeze()
        ma_pair_not_bond = check_existence(graph.edge_index.t(), torch.cat([graph.bond_pair[graph.bonded_pair.squeeze(dim=-1)  == 0, :], graph.bond_pair[graph.bonded_pair.squeeze(dim=-1)  == 0, :][:, [1, 0]]])).squeeze()
        bf = torch.zeros(graph.num_edges)
        bf[ma_pair_is_bond == 1] = 1
        bf[ma_pair_not_bond == 1] = 2
        bf = torch.nn.functional.one_hot(bf.long(), num_classes=3).float()
        bf = bf[:, 1:]
        graph.edge_attr = torch.cat([graph.edge_attr, bf], dim=-1)
    # add target y
    if targets is not None:
        y = torch.Tensor([targets])
        graph.y = y
    if nodey is not None:
        nodey = torch.Tensor(nodey)
        graph.nodey = nodey
    if edgey is not None:
        edgey = torch.Tensor(edgey)
        graph.edgey = edgey

    return graph


def grab_element_feature(
    elements=['Mo', 'Mn', 'Fe', 'Ru', 'Co', 'Ni', 'Pd', 'Cu', 'N', 'H'],
    features='All',
    normalization=None,
    use_onehot=False,
    onehot_dim=6,
    compress_onehot=False,
    symbol_index=False
):
    '''
    Function to grab element features from pymatgen.core.Element
    
    Args:
        - elements (list): Default = ['Mo', 'Mn', 'Fe', 'Ru', 'Co', 'Ni', 'Pd', 'Cu', 'N', 'H']
        - features (list or 'All'): select element features. Default = 'All'
          Choose from 'number'(atomic number), 'row', 'group', 'atomic_radius', 'atomic_mass', 'Molar volume', 
          'X'(electronegativity), 'electron_affinity', 'ionization_energy', 'valence', 'Atomic orbitals'(energy of HOMO)
        - normalization (str or None): Normalize numerical features ('min-max', 'z-score', 'max-abs'). Default = None 
        - use_onehot (bool): Turn all features into one-not features. Default = False
        - onehot_dim (int): dimension of numerical turned one-hot features. Default = 6
        - compress_onehot (bool): compress numerical turned one-hot features. Default = False
        - symbol_index (bool): use symbol as the indexes of df. Default = False 
    Return:
        - The element features with atomic numbers as indices (pd.DataFrame).
    '''
    # init
    disperse = ['number', 'row', 'group', 'valence']
    if features == 'All':
        all_features = ['number', 'row', 'group', 'atomic_radius', 'atomic_mass', 'Molar volume', 'X', 'electron_affinity', 'ionization_energy', 'valence', 'Atomic orbitals']
    else:
        all_features = features
    if use_onehot: 
        normalization = 'min-max'
    data = {}
    # loop 
    for feature in all_features:
        feature_array = np.zeros(len(elements))
        # get feature array
        for i, element in enumerate(elements):
            if   feature == 'Molar volume':
                feature_array[i] = float(Element(element).data['Molar volume'].split(' ')[0])
            elif feature == 'valence':
                valence = getattr(Element(element), 'electronic_structure')
                if   'd' in valence:
                    feature_array[i] = int(valence.split('d')[1].split('.')[0])
                elif 'p' in valence:
                    feature_array[i] = int(valence.split('p')[1].split('.')[0])
                elif 's' in valence:
                    feature_array[i] = int(valence.split('s')[1].split('.')[0])
            elif feature == 'Atomic orbitals':
                if   Element(element).group <= 2:
                    feature_array[i] = Element(element).data['Atomic orbitals'][str(Element(element).row) + 's']
                elif Element(element).group > 13:
                    feature_array[i] = Element(element).data['Atomic orbitals'][str(Element(element).row) + 'p']
                else:
                    feature_array[i] = Element(element).data['Atomic orbitals'][str(Element(element).row - 1) + 'd']
            else:
                feature_array[i] = getattr(Element(element), feature)
        # normalization
        if normalization is not None and feature not in disperse:
            all_negative = np.all(feature_array < 0)
            if   normalization == 'min-max':
                maxval = np.max(feature_array)
                minval = np.min(feature_array)
                if all_negative:
                    feature_array = - (feature_array - maxval) / (maxval - minval)
                else:
                    feature_array = (feature_array - minval) / (maxval - minval)
            elif normalization == 'z-score':
                mean = np.mean(feature_array)
                std  = np.std(feature_array)
                feature_array = (feature_array - mean) / std
            elif normalization == 'max-abs':
                if all_negative:
                    minabs = np.abs(np.min(feature_array))
                    feature_array = feature_array / minabs
                else:
                    maxabs = np.abs(np.max(feature_array))
                    feature_array = feature_array / maxabs
        # one-hot
        if use_onehot:
            feature_array = torch.Tensor(feature_array)
            if feature in disperse:
                unique_values, inverse_indices = torch.unique(feature_array, sorted=True, return_inverse=True)
                feature_matrix = torch.nn.functional.one_hot(inverse_indices, num_classes=len(unique_values)).numpy()
            else:
                class_indices = (feature_array * onehot_dim).long()
                class_indices[torch.where(class_indices == onehot_dim)[0]] = onehot_dim - 1
                if compress_onehot:
                    unique_values, inverse_indices = torch.unique(class_indices, sorted=True, return_inverse=True)
                    feature_matrix = torch.nn.functional.one_hot(inverse_indices, num_classes=len(unique_values)).numpy()
                else:
                    feature_matrix = torch.nn.functional.one_hot(class_indices, num_classes=onehot_dim).numpy()
                    unique_values = np.arange(0, onehot_dim -1)
        # append
        if use_onehot:
            for feature_dim in range(feature_matrix.shape[1]):
                data[feature + '-' + str(int(unique_values[feature_dim]))] = feature_matrix[:, feature_dim]
        else:
            data[feature] = feature_array
    # create df
    df_feature = pd.DataFrame(data)
    if symbol_index:
        df_feature.index = elements
    else:
        df_feature.index = [getattr(Element(element), 'number') for element in elements]

    return df_feature


def read_atom_mag(
    OC_path_adss,
    OC_path_slab=None,
    adss_seq=None,
    delta_mag=False,
    site_only=False,
    adsb_ele=['N', 'H'],
    abs_mag_out=False,
    abs_mag_fin=False,
    struc='POSCARoo',
    outcar='OUTCAR',
    para_infer_site={},
    sta_ele_mag=False,
    disable_tqdm=True,
    jupyter_tqdm=True,
):
    '''
    Read atomic magnetization data from OUTCAR file

    Parameters:
        - OC_path_adss (path): path to directions containing adsorption structure OUTCAR files
        - OC_path_slab (path): path to directions containing pure slab structure OUTCAR file. Default = None
        - adss_sqe (list or None): direction names for data query. If None, use all in the OC_path_adss. Default = None
        - delta_mag (bool): calculate the increase in atomic magnetic moment compared to a pure surface (need OC_path_slab). Default = False
        - site_only (bool): only store the atomic magnetic moment of adsorption site (need adsb_ele). Default = False
        - adsb_ele (list): list of elements in adsorbate. Default = ['N', 'H']
        - abs_mag_out (bool): Take the absolute value of the magnetic moment value extracted from OUTCAR. Default = False
        - abs_mag_fin (bool): Take the absolute value of the magnetic moment that returns at the end. Default = False
        - struc (str): structure file name. Default = 'POSCARoo'
        - outcar (str): OUTCAR file name. Default = 'OUTCAR'
        - para_infer_site (dict): parameters on infer_adsorption_site except adsb. Default = {}
        - sta_ele_mag (bool): retuan a dict with (magnetization, adsorption structure name) for each atom
        - disable_tqdm (bool). Default = True
        - jupyter_tqdm (bool). Default = True
    Return:
        - a DataFrame that stores magnetization data, and a dict stores magnetization data for each element if sta_ele_mag
    '''
    # init
    mag_data = []
    if sta_ele_mag:
        sta_dict = {}
    # get adss seq
    if   adss_seq is not None:
        adss_seq = [f.split('.')[0] for f in adss_seq]
    elif adss_seq is None:
        adss_seq = get_file_or_subdirection(OC_path_adss, 'dir')
    remove_list = []
    # loop for each adss
    tqdm = tqdm_notebook if jupyter_tqdm else tqdm_common
    with tqdm(adss_seq, disable=disable_tqdm) as pbar:
        for adss_name in pbar:
            # check outcar exist
            if not os.path.exists(os.path.join(OC_path_adss, adss_name, outcar)): 
                remove_list.append(adss_name)
                continue
            # read structure and mag for ads_slab
            adss = Structure.from_file(os.path.join(OC_path_adss, adss_name, struc))
            magd_adss = read_mag_outcar(os.path.join(OC_path_adss, adss_name, outcar), abs_mag_out)
            if delta_mag:
                if OC_path_slab is not None and not os.path.exists(os.path.join(OC_path_slab, adss_name.split('_')[0], outcar)):
                    remove_list.append(adss_name)
                    continue
                adss_dela = adss.copy()
                adss_dela.remove_species(adsb_ele)
                magd_slab = read_mag_outcar(os.path.join(OC_path_slab, adss_name.split('_')[0], outcar), abs_mag_out)
                for i, site in enumerate(adss):
                    if site.specie.symbol not in adsb_ele:
                        magd_adss[i] = magd_adss[i] - magd_slab[adss_dela.index(site)]
            if site_only:
                _, _, cont = infer_adsorption_site(adss, adsb_ele, **para_infer_site)
                magd_adss = [magd_adss[i] for i in range(len(adss)) if i in cont[1]]
                magd_ele  = [adss[i].specie.symbol for i in range(len(adss)) if i in cont[1]]
            if abs_mag_fin:
                magd_adss = [abs(ma) for ma in magd_adss]
            if sta_ele_mag:
                if not site_only:
                    magd_ele = [adss[i].specie.symbol for i in range(len(adss))]
                for ele, mag in zip(magd_ele, magd_adss):
                    if ele not in sta_dict.keys():
                        sta_dict[ele] = []
                    sta_dict[ele].append((mag, adss_name))
            mag_data.append(np.array(magd_adss))
    df = pd.DataFrame.from_dict({'atomic_mag': mag_data})
    for remove_name in remove_list:
        adss_seq.remove(remove_name)
    df.index = adss_seq
    if sta_ele_mag:
        return df, sta_dict
    else:
        return df


def get_bond_feature(
    path_initi, 
    path_relax, 
    adss_seq=None, 
    para_infer_site_initi={}, 
    para_infer_site_relax={}, 
    disable_tqdm=True, 
    jupyter_tqdm=True,
):
    '''
    The bonding features are obtained by comparing the optimized structure with the initial structure

    Parameters:
        - path_initi (path): path to the initial structures stored direction
        - path_relax (path): path to the relaxed structures stored direction
        - adss_seq (list or None): structure file names for data query. If None, use all in the path_relax. Default = None
        - para_infer_site_initi (dict): parameters for infer_adsorption_site used in initial structures. Default = {}
        - para_infer_site_relax (dict): parameters for infer_adsorption_site used in relaxed structures. Default = {}
        - disable_tqdm (bool). Default = True
        - jupyter_tqdm (bool). Default = True
    Return:
        -  a DataFrame that stores bonding features data
    '''
    # init
    bond_data = []
    # get_adss_seq
    if   adss_seq is not None:
        adss_seq = [s for s in adss_seq]
    elif adss_seq is None:
        adss_seq = [s for s in get_file_or_subdirection(path_relax, 'file')]
    # loop for each adss
    tqdm = tqdm_notebook if jupyter_tqdm else tqdm_common
    with tqdm(adss_seq, disable=disable_tqdm) as pbar:
        for adss_name in pbar:
            struc_initi = Structure.from_file(os.path.join(path_initi, adss_name))
            struc_relax = Structure.from_file(os.path.join(path_relax, adss_name))
            site_initi, _, cnet_initi = infer_adsorption_site(struc_initi, **para_infer_site_initi)
            site_relax, _, cnet_relax = infer_adsorption_site(struc_relax, **para_infer_site_relax)
            if all([site in site_initi for site in site_relax]):
                if len(site_initi) != 4:
                    cneti1 = list(set(cnet_initi[0])); cneti1.sort()
                    cneti2 = list(set(cnet_initi[1])); cneti2.sort()
                    bond_combinations = np.array(tuple(itertools.product(cneti1, cneti2)))
                elif len(site_initi) == 4:
                    bond_combinations = np.array(cnet_initi).transpose()
                    bond_combinations = bond_combinations[np.lexsort((bond_combinations[:, 1], bond_combinations[:, 0]))]
                bond_relaxed = np.array(cnet_relax).transpose()
                bond_combinations = bond_combinations[:, np.newaxis, :]
                bond_relaxed = bond_relaxed[np.newaxis, :, :]
                bond_matches = bond_combinations == bond_relaxed
                bond_matches = np.all(bond_matches, axis=-1)
                bond_feature = np.any(bond_matches, axis=-1)
                bond_feature = bond_feature.astype(int)[:, np.newaxis]
            else:
                bond_feature = None
                print('site change at ', adss_name)
            bond_data.append(bond_feature.reshape(-1, ))
    df = pd.DataFrame.from_dict({'bond': bond_data})
    df.index = adss_seq
    return df