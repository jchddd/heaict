import os
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
from torch_geometric.loader import DataLoader
from torch_geometric.utils  import dense_to_sparse
from hgcode.utility         import get_file_or_subdirection, find_indices, get_element_symbol
from hgcode.site            import infer_adsorption_site


def load_graph_from_database(
    database,
    targets=['target'],
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
        - targets (list): list of keys for target values in database. Default = ['target']
        - save (bool): save graphs to pt file. Default = False
        - pt_file (file str): name of the saved pt file. Default = r'./graphs.pt'
        - disable_tqdm (bool). Default = True
        - jupyter_tqdm (bool). Default = True
        - para_grab_ele_feature (dict): parameters for grab_element_feature. Default = {}
          Template: {'features': 'All', 'normalization': True, 'use_atomber_onehot': False}. If 'elements' is not passed, elements will be grad from database automatically.
        - para_construct_graph (dict): parameters for on construct_graph and infer_adsorption_site. Default = {}
          Template: {'cutoff_substrate': 3, 'cutoff_adsorbate': 1.3, 'cutoff_bond': 2.7, 'adsb': ['N', 'H'], 'forbidden_bonds': [['H', 'H']]}
          And other key-value pairs in infer_adsorption_site except 'slab', 'adsb', and 'cutoff'
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
            featy = [float(getattr(row, target)) for target in targets]
            graph = add_features(graph, df_feature, featy, **para_add_feature)
            # add to list
            graphs.append(graph)
    # save graphs to pt file
    if save:
        torch.save(graphs, pt_file)
    
    return graphs
    

def build_database(
    slabs_direction,
    target_csv=None,
    targets=['target'],
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
        - target_csv (file str or None): csv file that stores target values with slab file names as the index, None is not written targets. Default = None
        - targets (list): column names for targets in csv file. Default = ['target']
          Each target value is written individually to a key-value pair in the database row. Some bad keys like energy can not be used.
        - database (str, file): path to the created database file. Default = r'./slab.db'
        - disable_tqdm (bool). Default = True
        - jupyter_tqdm (bool). Default = True
        - adsorbate_elements (list): list of adsorbate elements for n_neighbor_slab construction. Default = ['N', 'H']
        - para_n_neib_slab (dict or None): parameters on get_n_neighbor_slab, used to convert slab to n_neighbor_slab. None is store origin structures. Default = {}
          Template: {'number_neighbors': 2, 'max_neighbor_number': 12, 'cutoff': 3}
        - para_infer_ads_site (dict): parameters on infer_adsorption_site, used to determine the number of atoms on adsorption site for n_neighbor_slab construction. Default = {}
          Template: {'adsb_excl': [], 'method': 'cutoff', 'cutoff': 2.7, defin_dist: 'dist'}
    '''
    # load target file
    if target_csv is not None:
        df_target = pd.read_csv(target_csv, index_col=0)
    # check if database exists, and connect it 
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
            if target_csv is not None:
                for target in targets:
                    target_values[target] = float(df_target.loc[slab, target])
            # get slab structure
            slab = Structure.from_file(os.path.join(slabs_direction, slab))
            if 'selective_dynamics' in slab.site_properties.keys():
                slab.remove_site_property('selective_dynamics')
            # get n beighbor slab
            if para_n_neib_slab is not None:
                site, coor, cnet = infer_adsorption_site(slab, adsorbate_elements, **para_infer_ads_site)
                slab = get_n_neighbor_slab(slab, len(site), **para_n_neib_slab)
            # write slab and target to database
            slab = AseAtomsAdaptor.get_atoms(slab)
            database.write(slab, **target_values)
            
            
def get_n_neighbor_slab(
    slab, 
    number_site_atoms, 
    number_neighbors=2, 
    adsorbate_elements=['H', 'N'], 
    max_neighbor_number=66,
    cutoff=3,
):
    '''
    Functin to find n neighbor for adsorbate and construct the corresponding slab.
    The first neighbor refers to the adsorption site itself.
    
    Parameters:
        - slab (pymatgen.core.Structure).
        - number_site_atoms (int): number of adsorption site atoms. Default = 3
        - number_neighbors (int): the n of n neighbors. Default = 2
        - adsorbate_elements (list). Default = ['H', 'N']
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
    for site in adsorbate_sites:
        substrate.append(site.specie, site.coords + vector_to_center_image, True)
    # find n neighbors 
    adsorbate_indices = find_indices(substrate.species, [Element(e) for e in adsorbate_elements])
    neighbor_slab_indices = adsorbate_indices
    source_atom_indices = adsorbate_indices
    for n in range(number_neighbors):
        if n == 0: max_neighbor_number_for_n = number_site_atoms
        else: max_neighbor_number_for_n = max_neighbor_number
        n_neighbors = find_neighbors(substrate, source_atom_indices, max_neighbor_number_for_n, cutoff=cutoff)
        source_atom_indices = n_neighbors
        neighbor_slab_addition = n_neighbors[np.isin(n_neighbors, neighbor_slab_indices, invert=True)]
        neighbor_slab_indices = np.concatenate((neighbor_slab_indices, neighbor_slab_addition))
    # construct neighbor slab
    substrate_indices = np.arange(0, len(substrate))
    remove_indices = substrate_indices[np.isin(substrate_indices, neighbor_slab_indices, invert=True)]
    substrate.remove_sites(remove_indices)
    
    return substrate


def find_neighbors(
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


def construct_graph(
    slab,
    cutoff_substrate=3,
    cutoff_adsorbate=1.3,
    cutoff_bond=2.7,
    adsb=['N', 'H'],
    forbidden_bonds=[['H', 'H']],
    add_distance=False,
    print_info=False,
    **para_infer_ads_site
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
        - add_distance (bool): whether to add interatomic_distance to edge feature
        - print_info (bool): whether to print bonding information or not. Default = False
        - **para_infer_ads_site: parameters for infer_adsorption_site except cutoff and adsb.
    Return:
        - The graph (torch_geometric.data.Data).
    '''
    # find adsorbate and adsorption site
    site, coor, cnet = infer_adsorption_site(slab, cutoff=cutoff_bond, adsb=adsb, **para_infer_ads_site)
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
    mask_substrate[cnet[0] + cnet[1], cnet[1] + cnet[0]] = False
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
    # construct graph data
    data = Data()
    data.edge_index = edge_index
    data.interatomic_distance = edge_weight if add_distance else None
    data.atomic_number = torch.Tensor(slab.atomic_numbers).long()
    data.atomic_type = atomic_type

    return data


def add_features(
    graph, 
    df_feature,
    targets,
    coordinate=[9, 12],
    distance_feature=True,
    distance_classes=None
):
    '''
    Function to add features (node and edge feature, node cluster and target) to a graph
    
    Parameters:
        - graph (torch_geometric.Data).
        - df_feature (pd.DataFrame): dataframe that store node features.
        - targets (list): target values stored in a list.
        - coordinate (list or None): coordinate numbers for surface and bulk. Default = [9, 12]
        - distance_feature (bool): add graph distance related features to node and edge. Default = True
        - distance_classes (int or None): manual setting number classes of node distance. Default = None
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
        if bond_type_judge > 3: # Intermolecular covalent bonds
            bond_type[edge_index] = 2
        elif bond_type_judge > 1: # Ionic bonds between molecules and metals
            bond_type[edge_index] = 1
        else: # Metal bonds between metals
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
    if distance_feature:
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
    # add target y
    y = torch.Tensor([targets])
    graph.y = y

    return graph


def grab_element_feature(
    elements=['Mo', 'Mn', 'Fe', 'Ru', 'Co', 'Ni', 'Pd', 'Cu', 'N', 'H'],
    features='All',
    normalization=True,
    use_atomber_onehot=False,
    show_correlation_matrix=False
):
    '''
    Function to grab element features from pymatgen.
    
    Args:
        - elements (list). Default = ['Mo', 'Mn', 'Fe', 'Ru', 'Co', 'Ni', 'Pd', 'Cu', 'N', 'H']
        - features (list or 'All'): select element features. Default = 'All'
        - normalization (bool). Default = True
        - use_atomber_onehot (bool), use one-not feature of atomic number as feature. Default = False
        - show_correlation_matrix (bool). Default = False
    Return:
        - The element features with atomic numbers as indices (pd.DataFrame).
    '''
    # init
    if use_atomber_onehot:
        feature_matrix = np.eye(len(elements))
        atomic_numbers = [getattr(Element(element), 'number') for element in elements]
        all_features = atomic_numbers
    else:
        if features == 'All':
            all_features = ['row', 'group', 'atomic_radius', 'atomic_mass', 'Molar volume', 'electronegativity', 'electron_affinity', 'ionization_energy', 'valence', 'orbital']
        else:
            all_features = features
        atomic_numbers = np.zeros(len(elements), dtype=int)
        feature_matrix = np.zeros((len(elements), len(all_features)))
        # grab element features from pymatgen.core.Element
        for i, element in enumerate(elements):
            element = Element(element)
            atomic_numbers[i] = getattr(element, 'number')
            feature_matrix[i][0] = getattr(element, 'row')
            feature_matrix[i][1] = getattr(element, 'group')
            feature_matrix[i][2] = getattr(element, 'atomic_radius')
            feature_matrix[i][3] = getattr(element, 'atomic_mass')
            feature_matrix[i][4] = float(element.data['Molar volume'].split(' ')[0])
            feature_matrix[i][5] = getattr(element, 'X')
            feature_matrix[i][6] = getattr(element, 'electron_affinity')
            feature_matrix[i][7] = getattr(element, 'ionization_energy')
            valence = getattr(element, 'electronic_structure')
            if   'd' in valence:
                feature_matrix[i][8] = float(valence.split('d')[1].split('.')[0])
            elif 'p' in valence:
                feature_matrix[i][8] = float(valence.split('p')[1].split('.')[0])
            elif 's' in valence:
                feature_matrix[i][8] = float(valence.split('s')[1].split('.')[0])
            if   element.group <= 2:
                feature_matrix[i][9] = element.data['Atomic orbitals'][str(element.row) + 's']
            elif element.group > 13:
                feature_matrix[i][9] = element.data['Atomic orbitals'][str(element.row) + 'p']
            else:
                feature_matrix[i][9] = element.data['Atomic orbitals'][str(element.row - 1) + 'd']
        # process normalization
        if normalization:
            mean_val = np.mean(feature_matrix, axis=0)
            std_val = np.std(feature_matrix, axis=0)
            feature_matrix = (feature_matrix - mean_val) / std_val
        # plot correlation matrix
        if show_correlation_matrix:
            correlation_matrix = np.corrcoef(feature_matrix, rowvar=False)
            cmap = plt.cm.RdYlGn_r
            plt.figure(figsize=(8, 6))
            cax = plt.matshow(correlation_matrix, cmap=cmap)
            plt.colorbar(cax)
            plt.xticks(range(len(all_features)), all_features, rotation=90, ha='center')
            plt.yticks(range(len(all_features)), all_features)
            plt.title('Correlation Matrix Heatmap')
            plt.show()
    # generate DataFrame
    data = {}
    for i, feature in enumerate(all_features):
        data[feature] = feature_matrix[:, i]
    df_feature = pd.DataFrame(data)
    df_feature.index = atomic_numbers
    
    return df_feature