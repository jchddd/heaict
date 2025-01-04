from pymatgen.core import Structure, Element
from hgcode.dataset import get_n_neighbor_slab, construct_graph, add_features, grab_element_feature
from hgcode.utility import get_file_or_subdirection, get_function_parameter_name, create_reverse_dict, check_array1_contained_in_array2
from hgcode.site import infer_adsorption_site
from ase.visualize import view
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np
import matplotlib.pyplot as plt
import os
from torch_geometric.loader import DataLoader

import torch

from torch_geometric.utils import to_dense_batch



class brute_force_surface():
    def __init__(
        self,
        surface_size=(30, 30),
        n_neighbors=2,
        slab_elements=['Mo', 'Mn', 'Fe', 'Ru', 'Co', 'Ni', 'Pd', 'Cu', 'N', 'H'],
        adsorbate_elements=['N', 'H']
    ):
        self.surface_size = surface_size
        self.n_rows, self.n_colums = surface_size
        self.n_layers = n_neighbors
        self.periodic_boundary = [self.n_rows, self.n_colums, self.n_layers + 1]
        self.slab_elements = slab_elements
        self.adsorbate_elements = adsorbate_elements
        
        self.df_feature = grab_element_feature(slab_elements)
        

    def read_archetype(self, file_path, zero_frac, grid_basis_in_frac, **kwargs):
        '''
        Read archetype adsorption structure information from POSCARs.
        
        Parameters:
            - file_path (path): the path where archetype POSCARs are stored.
            - zero_frac ((3)-array like): fractional coordinates of atoms in relative position.
            - grid_basis_in_frac ((3, 3)-array like): the vector corresponding to the grid base vector in the fractional coordinate of archetype slab.
            - **kwargs: other key parameters for construct_graph and infer_adsorption_site except slab.
        Modifies:
            - Store archetype information in self.archetype_xxxx(graph, slab, maska, site)
        Example:
            - read_archetype(r'hgcode/adsb-site', (0.444, 0.444, 0.3), [(0.111, 0, 0), (0, 0.111, 0), (-0.111/3, -0.222/3, -0.099)])
        Cautions:
            - The POSCARs must follow the following naming method, start with a uniform adsorbate name for each adsorbate, end with .vasp, like 'N2_fcc_1.vasp'.
            - Use an identical slab structure to generate n_neighbor slabs to ensure accurate transformation of coordinates.
            - In the grid coordinates, Positive X is to the right, Positive Y is upwards, Positive Z is into the screen.
        '''
        self.archetype = {}
        self._map_archi_to_archn = {}
        archetypes = get_file_or_subdirection(file_path)
        
        for index, archetype in enumerate(archetypes):
            archetype_name = archetype.split('.')[0]
            self._map_archi_to_archn[index] = archetype_name
            archetype_slab = Structure.from_file(os.path.join(file_path, archetype))
            
            slab_graph = construct_graph(archetype_slab, **{k: v for k, v in kwargs.items() if k in get_function_parameter_name(construct_graph)})
            slab_graph = add_features(slab_graph, self.df_feature)
            
            archetype_site, _, _ = infer_adsorption_site(archetype_slab, **{k: v for k, v in kwargs.items() if k in get_function_parameter_name(infer_adsorption_site)})
            zero_index = np.argmin(np.sum(np.abs(archetype_slab.frac_coords - np.array(zero_frac)), axis=-1))
            grid_slab, mask_adsb = self._fractional_coordinate_to_grid(archetype_slab, zero_index, grid_basis_in_frac)
            
            grid_site = grid_slab[[int(site.split('-')[0]) for site in archetype_site]]
            grid_site = grid_site[np.argsort(np.sum(np.abs(grid_site), axis=-1))][:, :2]
            
            self.archetype[archetype_name] = {'graph': slab_graph, 'slab': grid_slab, 'site': grid_site, 'mask_adsb':mask_adsb}
        
        self._map_adsbn_to_archi = self._map_adsorbate_to_archetype_index()
        self._map_archi_to_adsbn = create_reverse_dict(self._map_adsbn_to_archi)

    def _fractional_coordinate_to_grid(self, slab, zero_index, grid_basis_in_frac, adsorbate_elements=['N', 'H']):
        '''
        Turn fractional coordinate to grid coordinate, and mask adsorbate atoms. Apply in read_archetype.
        '''
        grid_basis_vector = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        transformation_matrix = np.linalg.inv(np.array(grid_basis_in_frac).T).dot(grid_basis_vector)

        grid_vector = []
        mask_adsb = []
        for atom in slab:
            if atom.specie.symbol not in adsorbate_elements:
                grid_vector.append(np.round(transformation_matrix.dot(atom.frac_coords - slab[zero_index].frac_coords)).astype(np.int32))
                mask_adsb.append(False)
            else:
                grid_vector.append(np.array([0, 0, 0]))
                mask_adsb.append(True)

        return np.array(grid_vector), np.array(mask_adsb)
        
    def _map_adsorbate_to_archetype_index(self):
        '''
        Enumerate archetype names, return a dict which map adsorbate names to archetype index in self.archetype.keys(). Apply in Apply in read_archetype.
        '''
        adsorbate_index = {}
        
        for index, key in enumerate(list(self.archetype.keys())):
            adsorbate = key.split('_')[0]
            adsorbate_index.setdefault(adsorbate, []).append(index)
            
        for adsorbate, indexes in adsorbate_index.items():
            adsorbate_index[adsorbate] = np.array(indexes)
            
        return adsorbate_index


    def create_slab(
        self,
        composition = {'A': {'Mo': 0.5, 'Ru': 0.5}, 'B': {'Cu': 1}},
        A_vector = [[2, 2], [1, 0]]
    ):
        '''
        Function to create slab grid with shape (n_rows, n_columns, n_layers), and values fill by atomic number according to composition.
        
        Parameters:
            - composition (dict): composition for A and B sites. Default = {'A': {'Mo': 0.5, 'Ru': 0.5}, 'B': {'Cu': 1}}
            - A_vector ((2, 2) list): array used to distinguish A site. Default = [[2, 2], [1, 0]]
        '''
        slab = np.zeros((self.n_rows, self.n_colums, self.n_layers), dtype=int)
        rows, colums, layers = np.indices(slab.shape)
        
        slab_index = np.stack((rows, colums, layers), axis=-1)
        slab_zeroA = self._A_to_zero(slab_index, A_vector)
        
        slab_maskA = slab_zeroA == 0
        number_A = np.count_nonzero(slab_maskA)
        slab_maskB = slab_zeroA != 0
        number_B = np.count_nonzero(slab_maskB)
        
        site_A_atoms = self._get_site_atomic_numbers(composition['A'], number_A)
        slab[np.where(slab_maskA)] = site_A_atoms
        site_B_atoms = self._get_site_atomic_numbers(composition['B'], number_B)
        slab[np.where(slab_maskB)] = site_B_atoms
        
        self.slab = slab
                    
    def _A_to_zero(self, slab_index, A_vector):
        '''
        Calculates an array used to distinguish A, B sites according to A_vector, A sites will be 0. Apply in create_slab.
        '''
        def vector_check(index_array):
            return np.sum((index_array + index_array[-1] * np.array(A_vector[1] + [0])) % np.array(A_vector[0] + [1]))
        return np.apply_along_axis(vector_check, axis=-1, arr=slab_index)
    
    def _get_site_atomic_numbers(self, composition, total_number):
        '''
        Get a vector with random distributed atomic numbers according to their ration in composition. Apply in create_slab.
        '''
        element_numbers = np.round([total_number * ration for ration in composition.values()]).astype(np.int32)
        difference = total_number - np.sum(element_numbers)
        element_numbers[-1] += difference
        
        atomic_numbers = [Element(element).Z for element in composition.keys()]
        
        site_atomic_numbers = np.array([atomic_number for atomic_number, repeat in zip(atomic_numbers, element_numbers) for _ in range(repeat)])
        np.random.shuffle(site_atomic_numbers)
        
        return site_atomic_numbers
                    
                    
    def eval_gross_energies(self, model, classify=True, save_indexes=False):
        '''
        Evaluate gross adsorption energies for each adsorbate-site pairs by a machine learning model.
        
        Parameters:
            - model (torch.model): the machine learning model.
            - classify (bool): whether the predictions including classification information or not. Default = True
            - save_indexes (bool): whether to save atom indexes or not. Default = False
        Modifies:
            - Store energies for each adsorbate-site pairs on self.gross_energies as an numpy.array with a shape of (number_pairs, nrows, ncolumns).
        '''
        number_archetype = len(self.archetype.keys())
        self.gross_energies = np.zeros((number_archetype, self.n_rows, self.n_colums))
        if save_indexes: self.slab_indexes = {}
        
        for i_archetype, archetype in enumerate(self.archetype.keys()):
            graph_list = []
            slab_indexes = []
            
            grid_slab = self.archetype[archetype]['slab']
            mask_adsb = self.archetype[archetype]['mask_adsb']
            graph_archetype = self.archetype[archetype]['graph']
            atomic_number_archetype = graph_archetype.atomic_number.numpy()
            
            for row in range(self.n_rows):
                for column in range(self.n_colums):
                    slab_center = [row, column, 0]
                    slab_indexes.append(((grid_slab + slab_center) % self.periodic_boundary).T)
            if save_indexes: self.slab_indexes[archetype] = slab_indexes
                    
            slab_indexes = np.concatenate(slab_indexes, axis=-1)
            slabs_labels = self.slab[tuple(slab_indexes)]
            slabs_labels = slabs_labels.reshape(-1, len(grid_slab))
            slabs_labels[:, mask_adsb] = atomic_number_archetype[mask_adsb]
            
            for slab_labels in slabs_labels:
                slab_graph = graph_archetype.clone()
                slab_graph.atomic_number = torch.Tensor(slab_labels).long()
                slab_graph.x = torch.vstack([torch.Tensor(self.df_feature.loc[atomic_number, :].values) for atomic_number in slab_labels])
                graph_list.append(slab_graph)
                
            energies = self._predict_gross_energies(graph_list, model, classify)
            self.gross_energies[i_archetype] = energies
            
    def _predict_gross_energies(self, graph_list, model, classify):
        '''
        Predict adsorption energies according to a graph list by a machine learning model. Apply in eval_gross_energies.
        '''
        model.eval()
        dataloader = DataLoader(graph_list, batch_size=len(graph_list), shuffle=False)#, num_workers=4, pin_memory=True)
        with torch.no_grad():
            for input_data in dataloader:
                predictions = model(input_data)
        if classify:
            energies = predictions[0][:, 0].reshape([10, 10]).numpy()
        return energies
    

    def plot_surface(self, apply_boundary=False):
        vector = np.array([[1, 0], [-0.5, 0.866], [0, -0.866*2/3]])
        
        color_dict = {'Mo':'purple', 'Fe':'r', 'Ru':'skyblue', 'Cu':'orange'}
        surface_zorder = 6
        for row in range(self.n_rows):
            for column in range(self.n_colums):
                for layer in range(self.n_layers):
                    atom_position = np.sum(vector * np.array([[row], [column], [layer]]), axis=0)
                    if apply_boundary:
                        boundary = np.array([self.n_rows, self.n_colums * 0.866]) ##################
                        atom_position = np.where(atom_position < 0, atom_position + boundary, atom_position)
                    atom_zorder = surface_zorder - layer
                    element = Element.from_Z(self.slab[row, column, layer]).symbol
                    plt.scatter(atom_position[0], atom_position[1], edgecolor='k', color=color_dict[element], zorder=atom_zorder, s=636)

                    
        
    def _get_site_atoms(self, anchor):
        '''
        Input a site anchor as (archetype, n_row, n_column), return site atoms of this site (atoms, n_rows, n_columns). Apply in get_net_energies.
        '''
        return (self.archetype[self._map_archi_to_archn[anchor[0]]]['site'] + anchor[1:]) % self.periodic_boundary[:2]

    def _search_min_e_from_reactions(self, reaction_steps):
        '''
        Search minest energy from adsorbates in the first reaction step. Apply in get_net_energies.
        '''
        reaction_steps = reaction_steps[:, 0]
        search_archis = np.concatenate([self._map_adsbn_to_archi[adsorbate] for adsorbate in reaction_steps])

        min_e_value = np.min(self.net_energies[search_archis])
        anchor_grid = np.unravel_index(self.net_energies[search_archis].argmin(), self.net_energies[search_archis].shape)
        anchor_grid = (search_archis[anchor_grid[0]], anchor_grid[1], anchor_grid[2])

        reaction_id = np.where(reaction_steps == self._map_archi_to_adsbn[anchor_grid[0]])[0][0]

        return min_e_value, anchor_grid, reaction_id

    def _search_min_e_with_shared_atom(self, search_adsb, atom_ensemble):
        '''
        Search minest energy for a adsorbate with sites cover or inside an ensemble of atoms (a site). Apply in get_net_energies.
        '''
        search_archis = self._map_adsbn_to_archi[search_adsb]

        search_anchors = []
        for archi in search_archis:
            search_anchors = self._find_anchor_of_sites_cover_inside_ensemble(archi, atom_ensemble)

        search_energies = [self.net_energies[anchor] for anchor in search_anchors]
        min_e_value = min(search_energies)
        anchor_atom = search_anchors[search_energies.index(min_e_value)]

        return min_e_value, anchor_atom

    def _find_anchor_of_sites_cover_inside_ensemble(self, archi, atom_ensemble):
        '''
        Find sites cover of inside an ensemble of atoms, return their anchors. Apply in get_net_energies.
        '''
        arch_site = self.archetype[self._map_archi_to_archn[archi]]['site']
        check_sites = self._enumerate_archsite_include_specified_atoms(arch_site, atom_ensemble)

        delete_ids = []
        for i, sitec in enumerate(check_sites):
            if not (check_array1_contained_in_array2(sitec, atom_ensemble, 'all') or check_array1_contained_in_array2(atom_ensemble, sitec, 'all')):
                delete_ids.append(i)
        check_sites = np.delete(check_sites, delete_ids, axis=0)

        anchors = [(archi, site[0][0], site[0][1]) for site in check_sites]

        return anchors

    def _block_anchors(self, atom_ensemble, block_way='site'):
        '''
        Block anchors corresponding to sites that share atom with an ensemble of atoms, or just block the input anchors. Apply in get_net_energies.
        '''
        block_anchors = []
        if   block_way == 'site':
            for archi, archn in enumerate(self.archetype.keys()):
                block_anchors += self._find_anchors_with_shared_atom(archi, archn, atom_ensemble)
        elif block_way == 'anchor':
            for archi, archn in enumerate(self.archetype.keys()):
                block_anchors += [(archi, site[1], site[2]) for site in atom_ensemble]
        self.net_energies[tuple(np.array(block_anchors).T)] = np.ma.masked

    def _enumerate_archsite_include_specified_atoms(self, archsite, atom_ensemble):
        '''
        Returns sites at which atoms in an ensemble are placed at each atom of one prototype site. Apply in get_net_energies.
        '''
        sites = []
        for atoma in archsite:
            for atome in atom_ensemble:
                sites.append(((archsite - atoma) + atome) % self.periodic_boundary[:2])
        sites = np.unique(np.array(sites), axis=0)
        return sites

    def _find_anchors_with_shared_atom(self, archi, archn, atom_ensemble):
        '''
        Find sites of an archtype that including atoms in an atom ensemble, return their anchors. Apply in get_net_energies.
        '''
        find_site_arch = self.archetype[archn]['site']
        check_sites = self._enumerate_archsite_include_specified_atoms(find_site_arch, atom_ensemble)

        delete_ids = []
        for i, site in enumerate(check_sites):
            if not check_array1_contained_in_array2(site, atom_ensemble, 'any'):
                delete_ids.append(i)
        check_sites = np.delete(check_sites, delete_ids, axis=0)

        anchors = [(archi, site[0][0], site[0][1]) for site in check_sites]

        return anchors

    def get_net_energies(
        self, 
        reaction_steps=np.array([['N2', 'NNH', 'NH', 'NH3'],['NNH', 'N2', 'NH', 'NH3'],['H', None, None, None]])
    ):
        sss = []
        
        try: isinstance(self.gross_energies, dict)
        except: self.get_gross_energies()
        self.net_energies      = np.ma.masked_array(np.copy(self.gross_energies), mask=np.zeros(self.gross_energies.shape))
        self.adsorption_site   = np.zeros(self.gross_energies.shape).astype(bool)

        while True:
            min_e = np.inf
            reaction_energies = []
            reaction_anchors  = []

            min_e_value, anchor_grid, reaction_id = self._search_min_e_from_reactions(reaction_steps)
            if np.ma.is_masked(min_e_value): break
            reaction_energies.append(min_e_value)
            reaction_anchors.append(anchor_grid)
            site_ensemble = self._get_site_atoms(anchor_grid)

            reaction_path = reaction_steps[reaction_id]
            for adsb in reaction_path[1:]:
                continue_while = False
                if   adsb is not None:
                    min_e_value, anchor_grid = self._search_min_e_with_shared_atom(adsb, site_ensemble)
                    if np.ma.is_masked(min_e_value):
                        self._block_anchors(reaction_anchors, 'anchor')
                        continue_while = True; break
                    else:
                        reaction_energies.append(min_e_value)
                        reaction_anchors.append(anchor_grid)
                        site_ensemble = np.unique(np.concatenate([site_ensemble, self._get_site_atoms(anchor_grid)]), axis=0)
                elif adsb is None:
                    break

            if continue_while: continue
            self.adsorption_site[tuple(np.array(reaction_anchors).T)] = True
            self._block_anchors(site_ensemble, 'site')

            print([round(e, 2) if e is not None and e < 1 else 'N' for e in reaction_energies ], reaction_anchors, reaction_path)
            sss.append(site_ensemble)
        return sss