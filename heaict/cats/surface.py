from heaict.cats.utility import *
from heaict.data.dataset import get_n_neighbor_slab, construct_graph, add_features, grab_element_feature
from heaict.data.site import infer_adsorption_site

from torch_geometric.loader import DataLoader
from pymatgen.core import Structure, Element
from scipy.spatial.distance import cdist
from multiprocessing import cpu_count
import numpy as np
import torch
import os


class brute_force_surface():
    '''
    Class for evaluation of catalytic properties on alloy surfaces.

    Methods:
        - read_archetype: Read archetype adsorption structures information from POSCARs.
        - create_slab_HEA: Function to create HEA slab grid, and fill values by atomic number according to composition.
        - create_slab_HEI: Function to create HEI slab grid, and fill values by atomic number according to composition.
        - eval_gross_energies: Evaluate gross adsorption energies for each adsorbate-site pairs by a machine learning model.
        - get_net_energies: Predict net energies distribution of the surface.
        - get_max_coverage: Get the maximum coverage of the surface.
        - count_site: Return adsorption site type (element composition) and corresponding site numbers.
        - plot_surface: Draw a schematic diagram of the surface atomic arrangement and adsorption sites.
    '''
    def __init__(
        self,
        surface_size=(30, 30),
        n_neighbors=3,
        slab_elements=['Mo', 'Mn', 'Fe', 'Ru', 'Co', 'Ni', 'Pd', 'Cu', 'N', 'H'],
        adsorbate_elements=['N', 'H'],
        para_grab_feature={}
    ):
        '''
        Parameters:
            - surface_size ((2, ) tuple): size of surface (/atom). Default = (30, 30)
            - n_neighbors (int): n_neighbors / n_layers of the slab. Default = 2
            - slab_elements (list of str): all elements on slabs (including adsorbate) for node feature generation.
              Default = ['Mo', 'Mn', 'Fe', 'Ru', 'Co', 'Ni', 'Pd', 'Cu', 'N', 'H']
            - adsorbate_elements (list of str): adsorbate elements. Default = ['N', 'H']
            - para_grab_feature (dict): parameters for grab_element_feature. Default = {}
        '''
        self.surface_size = surface_size
        self.n_x, self.n_y = surface_size
        self.n_layers = n_neighbors
        self.periodic_boundary = [self.n_x, self.n_y, self.n_layers]
        self.adsorbate_elements = adsorbate_elements

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_num_threads(cpu_count())
        
        self.df_feature = grab_element_feature(slab_elements, **para_grab_feature)
        self.mapping_eleTindex = {element: index for index, element in enumerate(self.df_feature.index)}
        self.feature_tensor = torch.Tensor(np.vstack([self.df_feature.loc[i, :] for i in self.df_feature.index])).to(self.device)
        

    def read_archetype(self, file_path, zero_frac, grid_basis_in_frac, para_constr_graph={}, para_add_feat={}, para_infer_site={}):
        '''
        Read archetype adsorption structures information from POSCARs.
        
        Parameters:
            - file_path (path): the path where archetype POSCARs are stored.
            - zero_frac ((3)-array like): fractional coordinates of atoms in relative position (origin/center/target point).
            - grid_basis_in_frac ((3, 3)-array like): grid base vector (x, y, z) in the fractional coordinate of archetype slab.
            - para_constr_graph (dict): other key parameters for construct_graph except slab.
            - para_add_feat (dict): other key parameters for add_features except slab and df_feature.
            - para_infer_site (dict): other key parameters for infer_adsorption_site except slab.
        Modifies:
            - Store archetype information in self.archetype_xxxx(graph, slab, maska, site)
        Example:
            - read_archetype(r'hgcode/adsb-site', (0.444, 0.444, 0.3), [(0.111, 0, 0), (0, 0.111, 0), (-0.111/3, -0.222/3, -0.099)])
        Cautions:
            - The POSCARs must follow the following naming method, start with a uniform adsorbate name for each adsorbate, end with .vasp, like 'N2_fcc_1.vasp'.
            - Use an identical slab structure to generate n_neighbor slabs to ensure accurate transformation of coordinates.
            - In the grid coordinates, Positive X is to the right, Positive Y is upwards, Positive Z is into the screen.
            - Atoms corresponding to the same x and y should have the same symmetry in the layer, like all A in an AB3 alloy.
        '''
        # init
        self.archetype = {}
        self._map_archi_to_archn = {}
        archetypes = get_file_or_subdirection(file_path)
        # loop for each archetype
        for index, archetype in enumerate(archetypes):
            # read archetype structure
            archetype_name = archetype.split('.')[0]
            self._map_archi_to_archn[index] = archetype_name
            archetype_slab = Structure.from_file(os.path.join(file_path, archetype))
            # construct slab graph
            slab_graph = construct_graph(archetype_slab, **para_constr_graph)
            slab_graph = add_features(slab_graph, self.df_feature, **para_add_feat)
            # get atom site and grid
            archetype_site, _, _ = infer_adsorption_site(archetype_slab, **para_infer_site)
            zero_index = np.argmin(np.sum(np.abs(archetype_slab.frac_coords - np.array(zero_frac)), axis=-1))
            grid_slab, mask_adsb = self._fractional_coordinate_to_grid(archetype_slab, zero_index, grid_basis_in_frac, self.adsorbate_elements)
            grid_site = grid_slab[[int(site.split('-')[0]) for site in archetype_site]]
            grid_site = grid_site[np.argsort(np.sum(np.abs(grid_site), axis=-1))][:, :2]
            # save archetype info
            self.archetype[archetype_name] = {'graph': slab_graph.to(self.device), 'slab': grid_slab, 'site': grid_site, 'mask_adsb': mask_adsb}
        # generate mapping between archetype names and indices
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

    def create_slab_HEA(
        self,
        composition = {'Mo': 0.5, 'Ru': 0.5}
    ):    
        '''
        Function to create slab grid with shape (n_x, n_y, n_layers), and fill values by atomic number according to composition.
        '''
        number_atom = self.n_x * self.n_y * self.n_layers
        site_atoms = self._get_site_atomic_numbers(composition, number_atom)
        slab = site_atoms.reshape((self.n_x, self.n_y, self.n_layers))
        self.slab = slab
    
    def create_slab_HEI(
        self,
        composition = {'A': {'Mo': 0.5, 'Ru': 0.5}, 'B': {'Cu': 1}},
        B_vector = [[2, 2], [1, 0]]
    ):
        '''
        Function to create slab grid with shape (n_x, n_y, n_layers), and fill values by atomic number according to composition.
        
        Parameters:
            - composition (dict): composition for A and B sites. Default = {'A': {'Mo': 0.5, 'Ru': 0.5}, 'B': {'Cu': 1}}
            - B_vector ((2, 2) list): array used to distinguish B site. Default = [[2, 2], [1, 0]]
        '''
        # init slab
        slab = np.zeros((self.n_x, self.n_y, self.n_layers), dtype=int)
        # creat atom index matrix
        rows, colums, layers = np.indices(slab.shape)
        slab_index = np.stack((rows, colums, layers), axis=-1)
        # generate site mask
        slab_zeroB = self._B_to_zero(slab_index, B_vector)
        slab_maskB = slab_zeroB == 0
        number_B = np.count_nonzero(slab_maskB)
        slab_maskA = slab_zeroB != 0
        number_A = np.count_nonzero(slab_maskA)
        # random element occupy
        site_B_atoms = self._get_site_atomic_numbers(composition['B'], number_B)
        slab[np.where(slab_maskB)] = site_B_atoms
        site_A_atoms = self._get_site_atomic_numbers(composition['A'], number_A)
        slab[np.where(slab_maskA)] = site_A_atoms
        # save slab
        self.slab = slab
                    
    def _B_to_zero(self, slab_index, B_vector):
        '''
        Calculates an array used to distinguish A, B sites according to A_vector, A sites will be 0. Apply in create_slab.
        '''
        def vector_check(index_array):
            return np.sum((index_array + index_array[-1] * np.array(B_vector[1] + [0])) % np.array(B_vector[0] + [1]))
        return np.apply_along_axis(vector_check, axis=-1, arr=slab_index)
    
    def _get_site_atomic_numbers(self, composition, total_number):
        '''
        Get a vector with random distributed atomic numbers according to their ration in composition. Apply in create_slab.
        '''
        element_numbers = np.round([total_number * ration for ration in composition.values()]).astype(np.int32)
        difference = total_number - np.sum(element_numbers)
        element_numbers[np.argmax(np.array(list(composition.values())))] += difference
        
        atomic_numbers = [Element(element).Z for element in composition.keys()]
        
        site_atomic_numbers = np.array([atomic_number for atomic_number, repeat in zip(atomic_numbers, element_numbers) for _ in range(repeat)])
        np.random.shuffle(site_atomic_numbers)
        
        return site_atomic_numbers
                    
    def eval_gross_energies(self, model, batch_size=2048, save_indexes=False, Ecor={}, use_round=None):
        '''
        Evaluate gross adsorption energies for each adsorbate-site pairs by a machine learning model.
        
        Parameters:
            - model (torch.model): the machine learning model.
            - batch_size (int): batch size. Default = 2048
            - save_indexes (bool): whether to save atom indexes of each slab or not. Default = False
            - Ecor (dict): Energy correction values for each archetype. Default = {} 
            - use_round (None or int): round the final gross energies. Default = None
        Modifies:
            - Store energies for each adsorbate-site pairs on self.gross_energies as an numpy.array with a shape of (number_pairs, nrows, ncolumns).
        '''
        # init
        model = model.to(self.device)
        number_archetype = len(self.archetype.keys())
        len_ele_feature  = self.feature_tensor.shape[1]
        self.gross_energies = np.zeros((number_archetype, self.n_x, self.n_y))
        if save_indexes: self.slab_indexes = {}
        # loop for each archetype
        for i_archetype, archetype in enumerate(self.archetype.keys()):
            # read archetype infor
            grid_slab = self.archetype[archetype]['slab']
            mask_adsb = self.archetype[archetype]['mask_adsb']
            graph_archetype = self.archetype[archetype]['graph']
            atomic_number_archetype = graph_archetype.atomic_number.cpu().numpy()            
            # loop for each atom, get archetype atom indices 
            slab_indexes = []
            for row in range(self.n_x):
                for column in range(self.n_y):
                    slab_center = [row, column, 0]
                    slab_indexes.append(((grid_slab + slab_center) % self.periodic_boundary).T)
            if save_indexes: self.slab_indexes[archetype] = slab_indexes
            # indices to symbol, add adorbate node
            slab_indexes = np.concatenate(slab_indexes, axis=-1)
            slabs_labels = self.slab[tuple(slab_indexes)]
            slabs_labels = slabs_labels.reshape(-1, len(grid_slab))
            slabs_labels[:, mask_adsb] = atomic_number_archetype[mask_adsb]
            # get node feature
            num_points, len_arche = slabs_labels.shape
            slabs_labels_all = slabs_labels.reshape(-1)
            slabs_labels_2index = np.array([self.mapping_eleTindex[ele] for ele in slabs_labels_all], dtype=int)
            slabs_x = self.feature_tensor[slabs_labels_2index, :].reshape(num_points, len_arche, -1)
            slabs_x = torch.Tensor(slabs_x)
            # generate torch graph data
            graph_list = []
            for i in range(num_points):
                slab_graph = graph_archetype.clone()
                # slab_graph.atomic_number = torch.Tensor(slab_labels).long()
                slab_graph.x[:, :len_ele_feature] = slabs_x[i]
                graph_list.append(slab_graph)
            # predict adsorption energy
            energies = self._predict_gross_energies(graph_list, model, batch_size)
            if archetype in Ecor: energies += Ecor[archetype]
            self.gross_energies[i_archetype] = energies
            # release gpu
            torch.cuda.empty_cache()
        if use_round is not None:
            self.gross_energies = np.round(self.gross_energies, use_round)
            
    def _predict_gross_energies(self, graph_list, model, batch_size):
        '''
        Predict adsorption energies according to a graph list by a machine learning model. Apply in eval_gross_energies.
        '''
        dataloader = DataLoader(graph_list, batch_size=batch_size, shuffle=False)
        predictions = self._get_predict(model, dataloader)[0]
        energies = predictions.reshape((self.n_x, self.n_y))
        return energies
    
    def _get_predict(self, model, loader):
        '''
        Input a model and a dataloader, return predicted values.
        '''
        model.eval()
        model.softmaxC = True
        # init
        predict = []
        predict_energies = []; predict_class1 = []; predict_class2 = []
        # predict
        with torch.no_grad():
            for batch in loader:
                predict_data = model(batch)    
                predict_energies.append(predict_data[0] if model.classify else predict_data) 
                if model.classify:
                    predict_class1.append(torch.argmax(predict_data[1], dim=-1))
                    predict_class2.append(torch.argmax(predict_data[2], dim=-1))
        # cat data
        predict_energies = torch.cat(predict_energies,dim=0).squeeze(-1).cpu().detach().numpy()
        if model.classify:
            predict_class1   = torch.cat(predict_class1  ,dim=0).squeeze(-1).cpu().detach().numpy()
            predict_class2   = torch.cat(predict_class2  ,dim=0).squeeze(-1).cpu().detach().numpy()
        # append
        predict.append(predict_energies)
        if model.classify:
            predict.append(predict_class1)
            predict.append(predict_class2)
        # return
        return predict
                    
    def _get_site_atoms(self, anchor):
        '''
        Input a site anchor as (archetype, n_row, n_column), return site atoms of this site (atoms, n_x, n_y). Apply in get_net_energies.
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
            search_anchors.extend(self._find_anchor_of_sites_cover_inside_ensemble(archi, atom_ensemble))

        search_energies = self.net_energies[tuple(np.array(search_anchors).T)]
        min_e_value = np.min(search_energies)
        anchor_atom = search_anchors[search_energies.argmin()]
        
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

    def _confirm_reaction_type(self, reaction_steps, barriers_calte):
        '''
        Confirm reaction types for each reaction step according to adsorbate types on barrier calculation. Apply in get_net_energies.
        '''
        reaction_types = []
        adsb_on_reatyp = {key: set([adsb for bar_step in value for adsb in bar_step if isinstance(adsb, str)]) for key, value in barriers_calte.items()}
        for reaction_step in reaction_steps:
            reaction_type = []
            for reaction in adsb_on_reatyp.keys():
                if adsb_on_reatyp[reaction] <= set(reaction_step):
                    reaction_type.append(reaction)
            reaction_types.append(reaction_type)
            
        return reaction_types

    def get_net_energies(
        self, 
        reaction_steps=[['N2', 'NNH', 'NH', 'NH3', 'H'], ['NNH', 'N2', 'NH', 'NH3', 'H'], ['H', 'N2', 'NNH', 'NH', 'NH3']],
        print_info=False
    ):
        '''
        Predict net energies of the surface

        Parameters:
            - reaction_steps ((n, r) list): n -> number of reaction. r -> maximum number of adsorbates in a single reaction.
              Here reaction refers to the sequence of seeking stable adsorbate structures in a common site. Use None to fill empty adsorbate
              Default = [['N2', 'NNH', 'NH', 'NH3', 'H'], ['NNH', 'N2', 'NH', 'NH3', 'H'], ['H', 'N2', 'NNH', 'NH', 'NH3']]
            - print_info: Pring information of each site, adsorption energy .etc. Default = False
        Return:
            - A dict with adsorption energies of each adsorbate on every sites
        '''
        if print_info: print('Adsorbates     -     Energies      -   Anchors')
        # find reaction type for each reaction_steps
        reaction_steps = np.array(reaction_steps)
        # create net_energies and adsorption site
        self.net_energies    = np.ma.masked_array(np.copy(self.gross_energies), mask=np.zeros(self.gross_energies.shape))
        self.adsorption_site = np.zeros(self.gross_energies.shape).astype(bool) # 0 -> False, != 0 -> True
        self.site_ensembles  = []
        self.site_anchors    = []
        self.site_adsenergy  = {adsb: [] for adsb in np.unique(reaction_steps[reaction_steps != None])}
        # start loop
        while True:
            # initialize variable 
            # min_e = np.inf
            step_energies = []
            step_anchors  = []
            # search the lowest energy in the first adsorbates of each reaction step 
            min_e_value, anchor_grid, reaction_id = self._search_min_e_from_reactions(reaction_steps)
            if np.ma.is_masked(min_e_value): break
            step_energies.append(min_e_value)
            step_anchors.append(anchor_grid)
            site_ensemble = self._get_site_atoms(anchor_grid)
            # search the lowest energies of other adsorbates on the reaction step
            reaction_path = reaction_steps[reaction_id]
            for adsb in reaction_path[1:]:
                continue_while = False
                if   adsb is not None:
                    min_e_value, anchor_grid = self._search_min_e_with_shared_atom(adsb, site_ensemble)
                    if np.ma.is_masked(min_e_value):
                        self._block_anchors(step_anchors, 'anchor')
                        continue_while = True; break
                    else:
                        step_energies.append(min_e_value)
                        step_anchors.append(anchor_grid)
                        site_ensemble = np.unique(np.concatenate([site_ensemble, self._get_site_atoms(anchor_grid)]), axis=0)
                elif adsb is None:
                    step_energies.append(np.nan)
            if continue_while: continue
            # block anchors corresponding to sites that have shared atoms with the current adsorption site
            self.adsorption_site[tuple(np.array(step_anchors).T)] = True
            self._block_anchors(site_ensemble, 'site')
            self.site_ensembles.append(site_ensemble)
            self.site_anchors.append(step_anchors)
            # store adsb adsorption energy into site_adsenergy
            for adsb in self.site_adsenergy:
                eads = step_energies[np.where(reaction_path==adsb)[0][0]] if adsb in reaction_path else np.nan
                self.site_adsenergy[adsb].append(eads)
            # print infomation
            if print_info:
                print(reaction_path, [round(e, 2) if e is not np.nan else 'None' for e in step_energies], step_anchors)
                
        # to numpy
        self.site_adsenergy = {key: np.array(value) for key, value in self.site_adsenergy.items()}
        # return
        if not hasattr(self, 'index'):
            return self.site_adsenergy
        else:
            return self.index, self.site_adsenergy
            
    def get_max_coverage(self):
        '''
        Get the maximum coverage of the surface.
        '''
        Nsite = 0
        for site in self.site_ensembles:
            Nsite += site.shape[0]
        return Nsite / (self.n_y * self.n_x)

    def count_site(self, grid_vector, indices=None):
        '''
        Return adsorption site type (element composition) and corresponding site numbers
    
        Parameters:
            - grid_vector ((3,3) np.array): grid dcar vector for 4-fold site identification.
            - indices (1D-array like int or None): indices of sites to be consider. Default = None
        Return:
            - a list of tuple including sites and numbers
            - for 4-hollow to set represent long and short diagonal element, for 3-hollow 3 elements are all listed
        '''
        if indices is None:
            site_ensembles = self.site_ensembles
        else:
            site_ensembles = [self.site_ensembles[i] for i in indices]
        
        unique_sites = []
        site_numbers = []
        
        for site in site_ensembles:
            site_archetype = site - site[0]
            site_withz = np.column_stack([site, np.zeros(site.shape[0], dtype=int).reshape(-1, 1)])
            x_idx = site_withz[:, 0]
            y_idx = site_withz[:, 1]
            z_idx = site_withz[:, 2]
            site_eles = self.slab[x_idx, y_idx, z_idx]
    
            site_set = None
            if   site_eles.shape[0] == 3:
                site_set = sort_by_frequency_unique(site_eles)
                if   len(site_set) == 1:
                    site_set = tuple(site_set * 3)
                elif len(site_set) == 2:
                    site_set = tuple([site_set[0]] * 2 + [site_set[1]])
                elif len(site_set) == 3:
                    site_set = set(site_set)
            elif site_eles.shape[0] == 4:
                site_archetype[site_archetype == self.n_y - 1] = -1
                site_archetype[site_archetype == self.n_y - 2] = -2
                site_archetype_with_z = np.column_stack([site_archetype, np.zeros(site_archetype.shape[0], dtype=int).reshape(-1, 1)])
                site_coords = site_archetype_with_z @ grid_vector  
                distance_matrix = cdist(site_coords, site_coords, metric='euclidean')
                max_distances = np.round(distance_matrix.max(axis=1), 2)
                long_diagonal = np.where(max_distances == np.max(max_distances))[0]
                short_diagonal = np.where(max_distances == np.min(max_distances))[0]
                site_set = [set(site_eles[long_diagonal]), set(site_eles[short_diagonal])]
        
            site_recored = False
            if len(unique_sites) > 0:
                for i, site in enumerate(unique_sites):
                    if np.array_equal(site_set, site):
                        site_numbers[i] += 1
                        site_recored = True
                        break
            if not site_recored:
                unique_sites.append(site_set)
                site_numbers.append(1)
        
        combined = list(zip(unique_sites, site_numbers))
        sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
        
        return sorted_combined

    def plot_surface(self, ax, color_dict, atomic_vector, x_range=None, y_range=None, n_layers=1, plot_site=True, site_indices=None, atom_edge={}, site_line={}):
        '''
        Draw a schematic diagram of the surface atomic arrangement and adsorption sites.

        Paremeters:
            - ax (matplotlib.axes)
            - color_dict (dict): color dict for element. like {'Cu': 'red', ...}
            - atomic_vector ((3, 3) array): The Cartesian coordinates of the atomic grid. 
              It can be obtained through the lattice constant @ atomic grid fractional coordinates. (grid_basis_in_frac)
            - x_range ((2, ) int tuple): The X-axis range of the plotted surface. Default = None
            - y_range ((2, ) int tuple): The Y-axis range of the plotted surface. Default = None
            - n_layers (int): Number of layers to plot. Default = 1
            - plot_site (bool): Show sites or not. (mainly used for the display of 3 and 4 fold hollow sites). Default = True
            - site_indices ((n, ) int list): Select the sites that need to be displayed. Default = None
            - atom_edge (dict): Parameters of atomic edges. Edge related parameters from plt.scatter. Default = {}
            - site_line (dict): Parameters of site edges. Edge related parameters from plt.plot. Default = {}
        '''
        if x_range is None:
            x_range = (0, self.n_x)
        if y_range is None:
            y_range = (0, self.n_y)
            
        atom_coords = []
        atom_symbols = []
        # plot substrate atoms
        topmost_surface_zorder = 6
        for x in range(*x_range):
            for y in range(*y_range):
                for layer in range(n_layers):
                    atom_position = np.array([x, y, layer]) @ atomic_vector #; atom_coords.append(atom_position)
                    atom_zorder = topmost_surface_zorder - layer
                    element = Element.from_Z(self.slab[x, y, layer]).symbol #; atom_symbols.append(element)
                    ax.scatter(atom_position[0], atom_position[1], color=color_dict[element], zorder=atom_zorder, **atom_edge)
        # plot adsorption sites
        if plot_site:
            if site_indices is None:
                site_ensembles = self.site_ensembles
            else:
                site_ensembles = [self.site_ensembles[i] for i in site_indices]
            for i, site in enumerate(site_ensembles):
                if all([np.all(x_range[0] < site[:, 0]), np.all(x_range[1] - 1 > site[:, 0]), np.all(y_range[0] < site[:, 1]), np.all(y_range[1] - 1 > site[:, 1])]):
                    site = sort_by_columns_lexicographically(site)
                    site_edge = [[], []]
                    for atom in site:
                        atom_position = np.array([atom[0], atom[1], 0]) @ atomic_vector
                        site_edge[0].append(atom_position[0])
                        site_edge[1].append(atom_position[1])
                    if len(site_edge[0]) == 4:
                        site_edge[0] = [site_edge[0][0], site_edge[0][1], site_edge[0][-1], site_edge[0][-2]]
                        site_edge[1] = [site_edge[1][0], site_edge[1][1], site_edge[1][-1], site_edge[1][-2]]
                    site_edge[0].append(site_edge[0][0])
                    site_edge[1].append(site_edge[1][0])
                    ax.plot(site_edge[0], site_edge[1], zorder=topmost_surface_zorder + 0.1, **site_line)
            
            ax.axis('off')
            ax.set_aspect('equal')

    
def get_activity_selectivity(site_adsenergy, Uspace=(-0.8, 0.2, 101), return_all=False, get_NRR_site_indices_at=None):
    '''
    Get activity and selectivity, coverage data from site_adsenergy

    Parameters:
        - site_adsenergy
        - Uspace ((3,) tuple): start and end U value as well as the number of U data points. Default = (-0.8, 0.2, 101)
        - return_all (bool): if not return the maxmimal V_NRR value and corresponding V_HER, FE and U. Default = False
          If true, use U_V_NRR, U_T_NRR, U_PDS_G1, U_PDS_G2, U_PDS_NRR, U_V_HER, U_T_HER, U_PDS_G3, U_PDS_G4, U_PDS_HER, U_FE = get_...
        - get_NRR_site_indices_at (float or None): get the indices of sites that N2 coverage is greater than H at the special U value. Default = None
    '''
    Uspace = np.linspace(*Uspace)
    if get_NRR_site_indices_at is not None:
        Uspace = [get_NRR_site_indices_at]
    U_V_NRR = []
    U_T_NRR = []
    U_PDS_G1 = []
    U_PDS_G2 = []
    U_PDS_NRR = []
    U_V_HER = []
    U_T_HER = []
    U_PDS_G3 = []
    U_PDS_G4 = []
    U_PDS_HER = []
    U_FE = []
    for U in Uspace:
        kb = 8.617e-5
        T  = 298.15
        # adsorption free energy for H2 and N2
        Gads_H = site_adsenergy['H'] + U
        Gads_N2 = site_adsenergy['N2']
        K_ads_H = np.exp(-Gads_H/(kb*T))
        K_ads_NRR = np.exp(-Gads_N2/(kb*T))
        # Coverage
        P = 1.0
        T_NRR = (K_ads_NRR * P) / (1 + K_ads_NRR * P + K_ads_H * P)
        T_HER = (K_ads_H * P) / (1 + K_ads_NRR * P + K_ads_H * P)
        # PDS for NRR
        G1 = site_adsenergy['NNH'] - site_adsenergy['N2'] + U
        G2 = site_adsenergy['NH3'] - site_adsenergy['NH'] + 2 * U
        PDS_NRR = np.max(np.vstack([G1, G2]), axis=0)
        PDS_NRR = np.where(T_NRR < T_HER, 1, PDS_NRR)
        # PDS for HER
        G3 = site_adsenergy['H'] + U
        G4 = -1 * site_adsenergy['H'] + U
        PDS_HER = np.max(np.vstack([G3, G4]), axis=0)
        # PDS to v
        PDS_HER = np.where(PDS_HER>0, PDS_HER, 0.0001)
        PDS_NRR = np.where(PDS_NRR>0, PDS_NRR, 0.0001)
        v_NRR = np.mean(T_NRR * np.exp(-PDS_NRR/(kb*T)))
        v_HER = np.mean(T_HER * np.exp(-PDS_HER/(kb*T)))
        # Faraday efficiency
        FE = (6 * v_NRR) / (6 * v_NRR + 2 * v_HER)
        
        U_V_NRR.append(v_NRR)
        U_T_NRR.append(np.mean(T_NRR))
        U_PDS_G1.append(np.mean(G1))
        U_PDS_G2.append(np.mean(G2))
        U_PDS_NRR.append(np.mean(PDS_NRR[np.where(T_NRR>=T_HER)[0]]) if len(np.where(T_NRR>T_HER)[0])>0 else -1)
        U_V_HER.append(v_HER)
        U_T_HER.append(np.mean(T_HER))
        U_PDS_G3.append(np.mean(G3))
        U_PDS_G4.append(np.mean(G4))
        U_PDS_HER.append(np.mean(PDS_HER))
        U_FE.append(FE)
        
    imax = np.argmax(np.array(U_V_NRR))
    if get_NRR_site_indices_at is not None:
        return np.where(T_NRR >= T_HER)[0]
    elif return_all:
        return U_V_NRR, U_T_NRR, U_PDS_G1, U_PDS_G2, U_PDS_NRR, U_V_HER, U_T_HER, U_PDS_G3, U_PDS_G4, U_PDS_HER, U_FE
    else:
        return U_V_NRR[imax], U_V_HER[imax], U_FE[imax], Uspace[imax]