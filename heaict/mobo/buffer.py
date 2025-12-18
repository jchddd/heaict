from heaict.mobo.utility import create_fibonacci_grid_on_sphere, find_neighbor_on_sphere_fibonacci
from heaict.mobo.utility import generate_weights_batch, get_edge_from_weights_batch
import numpy as np

class Buffer():
    '''
    Base class of performance buffer.

    Methods:
        - insert: Insert samples (X, Y) into buffers, which come from manifolds (patches) indexed by 'patch_ids'.
        - sample: Sample n samples in current buffers (close to the performance origin and distributed in different buffer units as much as possible).
        - k_nearest_sparse: sparsely approximate by sampling the K nearest samples on each buffer.
        - sparse_approximation: Use a few manifolds to sparsely approximate the Pareto front by graph-cut.
        - flattened: Return flattened x, y, and patch_id arrays from all the cells.
    Attribures:
        - n_obj (int): can only be 2 or 3
        - cell_num (int): number of discretized cells
        - cell_size (int): max sample number within each cell
        - origin ((n_obj, ) np.array): the origin point (minimum utopia)
        - sample_count (int): total number of samples
        - delta_b (float): normalization constaint for calculating unary energy in sparse approximation
        - cell_vertices ((~cell_num, n_var) np.array): Vector from the origin to the center of a uniformly distributed unit cell on the eighth half circle
        - cell_vertices_normalized ((~cell_num, n_var) np.array): normalized cell vertices
        - edges ((num_edges, 2) np.array): edges connecting those cell vertices (seen center on circle as nodes) 
        - buffer_x ((cell_num, ) list containing (num_sample_on_cell, n_var) np.array): x on each buffer
        - buffer_y ((cell_num, ) list containing (num_sample_on_cell, n_obj) np.array): y on each buffer
        - buffer_dist ((cell_num, ) list containing (num_sample_on_cell, ) list): distance to origin for samples on each buffer
        - buffer_patch_id ((cell_num, ) list containing (num_sample_on_cell, ) list): patch_ids for samples on each buffer
        - labels_opt ((num_cell, ) np.array): best patch_id (label) for each cell selected by graph-cut method
    '''
    def __init__(self, n_obj=2, cell_num=100, cell_size=10, delta_b=0.2, cell_shape='triangle'):
        '''
        Parameters:
            - n_obj (int): number of objectives. Default = 2
            - cell_num (int): number of discretized cells. Default = 100
            - cell_size (int): max sample number within each cell. Default = 10 
            - delta_b (float): normalization constaint for calculating unary energy in sparse approximation. Default = 0.2
            - cell_shape (str): type of 3D cell vertices, 'triangle' or 'sphere'. Default = 'triangle'
        '''
        self.n_obj = n_obj
        self.cell_num = cell_num
        self.cell_size = cell_size
        
        self.origin = np.zeros(n_obj)
        
        self.sample_count = 0

        self.delta_b = delta_b
        
        self.cell_shape = cell_shape
        self._initialize_cell()
        
    def _initialize_cell(self):
        if   self.n_obj == 2:
            thetas   = np.pi + (np.arange(self.cell_num) + 0.5) * (np.pi / 2 / self.cell_num)
            vertices = np.column_stack((1 + np.cos(thetas), 1 + np.sin(thetas)))
            edges    = np.array([[i, i + 1] for i in range(self.cell_num - 1)])
        elif self.n_obj == 3 and self.cell_shape == 'sphere':
            fibonacci_grid   = create_fibonacci_grid_on_sphere(self.cell_num * 8)
            quadrant_indices = np.where((fibonacci_grid[:, 0] <= 0) & (fibonacci_grid[:, 1] <= 0) & (fibonacci_grid[:, 2] <= 0))[0]
            vertices         = fibonacci_grid[quadrant_indices] + 1
            edges            = find_neighbor_on_sphere_fibonacci(vertices, quadrant_indices)
        elif self.n_obj == 3 and self.cell_shape == 'triangle':
            edge_cell_num = int(np.sqrt(2 * self.cell_num + 0.25) + 0.5) - 1
            vertices      = generate_weights_batch(n_dim=3, delta_weight=1.0 / (edge_cell_num - 1))
            edges         = get_edge_from_weights_batch(vertices)
            
        self.cell_vertices            = vertices
        self.cell_vertices_normalized = self.cell_vertices / np.linalg.norm(self.cell_vertices, axis=1)[:, np.newaxis]
        self.edges                    = edges
                    
    def insert(self, X, Y, patch_ids):
        '''
        Insert samples (X, Y) into buffers, which come from manifolds (patches) indexed by 'patch_ids'

        Parameters:
            - X ((num_sample, n_var) np.array)
            - Y ((num_sample, n_obj) np.array)
            - patch_ids ((num_sample,) list)
        '''
        has_move = self._move_origin(np.min(Y, axis=0))
        if self.sample_count == 0:
            self._initialize_buffer()
        elif self.sample_count > 0 and has_move:
            X_before, Y_before, patch_ids_before = self.flattened()
            X = np.vstack([X, X_before])
            Y = np.vstack([Y, Y_before])
            patch_ids = np.concatenate([patch_ids, patch_ids_before])
            self._initialize_buffer()
            
        F = Y - self.origin
        dists = np.linalg.norm(F, axis=1)
        cell_ids = self._find_cell_id(F)

        for x, y, cell_id, dist, patch_id in zip(X, Y, cell_ids, dists, patch_ids):
            self.buffer_x[cell_id].append(x)
            self.buffer_y[cell_id].append(y)
            self.buffer_dist[cell_id].append(dist)
            self.buffer_patch_id[cell_id].append(patch_id)
        self.sample_count += len(X)

        for cell_id in np.unique(cell_ids):
            self._update_cell(cell_id)
            
    def _find_cell_id(self, F):
        dots     = F @ self.cell_vertices_normalized.T
        cell_ids = np.argmax(dots, axis=1)
        return cell_ids

    def _move_origin(self, y_min):
        if (y_min >= self.origin).all() and not (y_min == self.origin).any():
            has_move = False
        else:
            self.origin = np.minimum(self.origin, y_min) - 1e-2
            has_move = True
        return has_move

    def _update_cell(self, cell_id):
        if len(self.buffer_dist[cell_id]) == 0: return
            
        idx = np.argsort(self.buffer_dist[cell_id])
        idx = idx[:self.cell_size]
        self.sample_count -= max(len(idx) - self.cell_size, 0)

        self.buffer_x[cell_id], self.buffer_y[cell_id], self.buffer_dist[cell_id], self.buffer_patch_id[cell_id] = \
            map(lambda x: list(np.array(x)[idx]), 
                [self.buffer_x[cell_id], self.buffer_y[cell_id], self.buffer_dist[cell_id], self.buffer_patch_id[cell_id]])
    
    def _initialize_buffer(self):
        self.buffer_x, self.buffer_y, self.buffer_dist, self.buffer_patch_id = [[[] for _ in range(self.cell_num)] for _ in range(4)]
        self.sample_count = 0
        
    def sample(self, n):
        '''
        Sample samples in current buffers (close to the performance origin and distributed in different buffer units as much as possible)

        Parameters:
            - n (int)
        Return:
            - selected_samples ((num_samples, n_var) np.array)
        '''
        selected_samples = []

        k = 0
        while k >= 0 and len(selected_samples) < n:
            nonempty_cell_ids = [i for i in range(self.cell_num) if len(self.buffer_dist[i]) > k]
            if len(nonempty_cell_ids) == 0:
                k = -1
            else:
                curr_selected_samples = [self.buffer_x[cell_ids][k] for cell_ids in nonempty_cell_ids]
                selected_samples.extend(np.random.permutation(curr_selected_samples))
                k += 1
        if len(selected_samples) < n:
            random_indices = np.random.choice(np.arange(len(selected_samples)), size=(n - len(selected_samples)))
            selected_samples = np.vstack([selected_samples, np.array(selected_samples)[random_indices]])
        selected_samples = np.array(selected_samples[:n])
        
        return selected_samples

    def k_nearest_sparse(self, K):
        '''
        sparsely approximate by sampling the K nearest samples on each buffer

        Parameter:
            - K (int)
        Returns:
            - K nearest X ((num_samples, n_var) np.array)
            - K nearest Y ((num_samples, n_obj) np.array)
        '''
        Knearest_X = []
        Knearest_Y = []

        k = 0
        while k >= 0:
            nonempty_cell_ids = [i for i in range(self.cell_num) if len(self.buffer_dist[i]) > k]
            curr_Knearest_X = [self.buffer_x[cell_ids][k] for cell_ids in nonempty_cell_ids]
            curr_Knearest_Y = [self.buffer_y[cell_ids][k] for cell_ids in nonempty_cell_ids]
            Knearest_X.extend(curr_Knearest_X)
            Knearest_Y.extend(curr_Knearest_Y)
            if k == K - 1:
                k = -1
            else:
                k += 1

        return np.array(Knearest_X), np.array(Knearest_Y)
            
    def sparse_approximation(self):
        '''
        Use a few manifolds to sparsely approximate the Pareto front by graph-cut.

        Returns:
            - labels ((n_label,) list): the optimized labels (manifold index) for each non-empty cell (the cells also contain the corresponding labeled sample)
            - approx_x ((n_label, n_var) np.array): the labeled design samples
            - approx_y ((n_label, n_obj) np.array): the labeled performance values
        '''
        from pygco import cut_from_graph
        # update patch ids, remove non-existing ids previously removed from buffer
        mapping = {}
        patch_id_count = 0
        for cell_id in range(self.cell_num):
            if self.buffer_patch_id[cell_id] == []: continue
            curr_patches = self.buffer_patch_id[cell_id]
            for i in range(len(curr_patches)):
                if curr_patches[i] not in mapping:
                    mapping[curr_patches[i]] = patch_id_count
                    patch_id_count += 1
                self.buffer_patch_id[cell_id][i] = mapping[curr_patches[i]]

        # construct unary and pairwise energy (cost) matrix for graph-cut
        C_inf = 10
        valid_cells = np.where([self.buffer_dist[cell_id] != [] for cell_id in range(self.cell_num)])[0]
        n_node = self.cell_vertices.shape[0]
        n_label = patch_id_count
        unary_cost = C_inf * np.ones((n_node, n_label))
        pairwise_cost = -6 * np.eye(n_label) # -C_inf
        
        for idx in valid_cells:
            patches, distances = np.array(self.buffer_patch_id[idx]), np.array(self.buffer_dist[idx])
            min_dist = np.min(distances)
            unary_cost[idx, patches] = np.minimum((distances - min_dist) / self.delta_b, C_inf)
        edges = self.edges
        
        # graph-cut by pygco
        edges, unary_cost, pairwise_cost = edges.astype(np.int32), unary_cost.astype(np.int32), pairwise_cost.astype(np.int32)
        labels_opt = cut_from_graph(edges, unary_cost, pairwise_cost)
        self.labels_opt = labels_opt

        # find corresponding design and performance values of optimized labels for each valid cell
        approx_xs, approx_ys = [], []
        labels = []
        for idx, label in zip(valid_cells, labels_opt[valid_cells]):
            for cell_patch_id, cell_x, cell_y in zip(self.buffer_patch_id[idx], self.buffer_x[idx], self.buffer_y[idx]):
                if cell_patch_id == label:
                    approx_xs.append(cell_x)
                    approx_ys.append(cell_y)
                    labels.append(label)
                    break
                else:
                    approx_xs.append(self.buffer_x[idx][0])
                    approx_ys.append(self.buffer_y[idx][0])
                    labels.append(label)
                    break
        approx_xs, approx_ys = np.array(approx_xs), np.array(approx_ys)
        return labels, approx_xs, approx_ys
        
    def flattened(self):
        '''
        Return flattened x, y, and patch_id arrays from all the cells

        Returns:
            - flattened_x: (total_sample_count, n_var) np.array
            - flattened_y: (total_sample_count, n_obj) np.array
            - flattened_patch_id: (total_sample_count, ) np.array
        '''
        flattened_x, flattened_y, flattened_patch_id = [], [], []
        for cell_x, cell_y, patch_id in zip(self.buffer_x, self.buffer_y, self.buffer_patch_id):
            if cell_x != []:
                flattened_x.append(cell_x)
                flattened_y.append(cell_y)
                flattened_patch_id.append(patch_id)
        return np.concatenate(flattened_x), np.concatenate(flattened_y), np.concatenate(flattened_patch_id)