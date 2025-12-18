from heaict.ml.SM_main import SurrogateModel
from heaict.mobo.problem import Real_Problem
from heaict.ml.GPR_scikit import GPR
from scipy.spatial import Delaunay
from scipy.stats   import qmc
from colorama      import Fore
import numpy       as np
import time


def search_feasible_grid(SP, lb=0, ub=1, interval=0.05):
    '''
    Search all the samples that meet the conclusion at certain intervals in the sample space

    Parameters:
        - SP (SurrogateModel or Real_Problem): on normalized or real X space respectively
        - lb (float): lower bound of samples. Default = 0
        - ub (float): upper bound of samples. Default = 1
        - interval (float): interval to create grid samples. Default = 0.05
    Return:
        - the grid ((n_feasibility, n_var) np.array)
    '''
    if   isinstance(SP, SurrogateModel):
        typ = 'SM'
        n_var = SP.real_problem.n_var
        feval = SP.evaluate_feasibility
    elif isinstance(SP, Real_Problem):
        typ = 'RP'
        n_var = SP.n_var
        feval = SP.evaluate_constraint_batch
    # create grid
    linspaces = []
    for i in range(n_var):
        linspaces.append(np.arange(lb, ub + interval, interval))
    grid = np.meshgrid(*linspaces, indexing='ij')
    grid = np.column_stack([g.reshape(-1) for g in grid])
    mask = feval(grid)
    if typ == 'RP':
        mask = np.all(mask <= 0, axis=1)
    grid = grid[mask]
    return grid
    
def print_with_timestamp(message, color='BLACK'):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(getattr(Fore, color) + f"{message} - [{timestamp}]")

def find_pareto_front(Y):
    '''
    Find pareto front (undominated part) of the input performance data.
    
    Parameters:
        - Y ((num_samples, n_obj) np.array)
    Returns:
        - pareto front ((num_pareto_points, n_obj) np.array)
        - pareto front indices ((num_pareto_points, ) list of int)
    '''
    if len(Y) == 0: return np.array([]), np.array([])

    sorted_indices = np.argsort(Y.T[0])
    pareto_indices = []
    for idx in sorted_indices:
        if not (np.logical_and((Y <= Y[idx]).all(axis=1), (Y < Y[idx]).any(axis=1))).any():
            pareto_indices.append(idx)
    pareto_front = Y[pareto_indices].copy()

    return pareto_front, pareto_indices

def generate_random_initial_samples(real_problem, n_sample, seed=None):
    '''
    Generate random initial samples by LHS method
    
    Parameters:
        - real_problem (Real_Problem)
        - n_sample (int): the number of samples to be generated
        - seed (int): the random seed
    Return:
        - random raw (not normalized) samples: ((n_sample, n_var) np.array)
    '''
    surrogate_model = GPR(real_problem)
    X_feasible = np.zeros((0, surrogate_model.real_problem.n_var))
    sampler = qmc.LatinHypercube(d=surrogate_model.real_problem.n_var, seed=seed)

    max_iter = 1000
    iter_count = 0
    while len(X_feasible) < n_sample and iter_count < max_iter:
        X_nor_samples = sampler.random(n=n_sample)
        feasible = surrogate_model.evaluate_feasibility(X_nor_samples)
        if np.any(feasible):
            X_feasible = np.vstack([X_feasible, X_nor_samples[feasible]])
        iter_count += 1
        
    if iter_count >= max_iter and len(X_feasible) < n_sample:
        raise Exception(f'hard to generate valid samples, {len(X_feasible)}/{n_sample} generated')

    X = X_feasible[:n_sample]
    return surrogate_model.normalization.undo(X)

def calculate_hypervolume(pareto_front, ref_point):
    '''
    Calculate the supervolume of the Pareto front

    Parameters:
        - pareto_front ((num_points, n_obj) np.array)
        - ref_point ((n_obj, ) np.array)
    Return:
        - the hypervolume (float)
    '''
    num_points = pareto_front.shape[0]
    num_objs = pareto_front.shape[1]
    hypervolume = 0

    # Sort the Pareto front according to the first target value
    sorted_front = pareto_front[pareto_front[:, 0].argsort()]

    def is_dominated(point, other_points):
        # Check whether other points dominate a point
        return np.any(np.all(other_points < point, axis=1))

    for i in range(num_points):
        point = sorted_front[i]
        if not is_dominated(point, sorted_front[:i]):
            sub_ref = ref_point.copy()
            for prev_point in sorted_front[:i]:
                for j in range(num_objs):
                    if prev_point[j] > point[j]:
                        sub_ref[j] = min(sub_ref[j], prev_point[j])
            volume = np.prod(sub_ref - point)
            hypervolume += volume
    return hypervolume

def generate_weights_batch(n_dim, delta_weight):
    '''
    Generate n dimensional uniformly distributed weights using depth first search.
    e.g. generate_weights_batch(2, 0.5) returns [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]

    Parameters:
        - n_dim (int): Dimension of the performance space.
        - delta_weight (float): Delta of the weight during generation.
    Returns:
       -  weights_batch (np.array):  Batch of weights generated.
    '''
    weights_batch = []
    generate_weights_batch_dfs(0, n_dim, 0.0, 1.0, delta_weight, [], weights_batch)
    return np.array(weights_batch)

def generate_weights_batch_dfs(i, n_dim, min_weight, max_weight, delta_weight, weight, weights_batch):
    if i == n_dim - 1:
        weight.append(1.0 - np.sum(weight[0:i]))
        weights_batch.append(weight.copy())
        weight = weight[0:i]
        return
    w = min_weight
    while w < max_weight + 0.5 * delta_weight and np.sum(weight[0:i]) + w < 1.0 + 0.5 * delta_weight:
        weight.append(w)
        generate_weights_batch_dfs(i + 1, n_dim, min_weight, max_weight, delta_weight, weight, weights_batch)
        weight = weight[0:i]
        w += delta_weight
        
def get_edge_from_weights_batch(vectors):
    vectors = trisurf_2D_T_3D(vectors)
    tri = Delaunay(vectors)
    ind, all_neighbors = tri.vertex_neighbor_vertices
    edges = []
    for i in range(len(vectors)):
        neighbors = all_neighbors[ind[i]:ind[i + 1]]
        for j in neighbors:
            edges.append(np.sort([i, j]))
    edges = np.unique(edges, axis=0)
    return edges

def trisurf_2D_T_3D(vectors):
    p1 = np.array([1, 0, 0])
    p2 = np.array([0, 1, 0])
    p3 = np.array([0, 0, 1])
    v1 = p3 - p1
    v2 = p2 - p1
    n = np.cross(v1, v2)
    n = n / np.linalg.norm(n)
    u1 = v1 / np.linalg.norm(v1)
    u2 = np.cross(n, u1)
    u2 = u2 / np.linalg.norm(u2)
    vectors_2d = np.zeros((vectors.shape[0], 2))
    for i, p in enumerate(vectors):
        vectors_2d[i, 0] = np.dot(p - p1, u1)
        vectors_2d[i, 1] = np.dot(p - p1, u2)
    return vectors_2d

def create_fibonacci_grid_on_sphere(num_points):
    def gen_fib_lattice(N):  
        phi = (1 + np.sqrt(5)) / 2  
        x   = (np.arange(1, N + 1) / phi) % 1  
        y   = np.arange(1, N + 1) / N  
        return x, y
    def area_preserve_rec2circ(x, y):  
        theta = 2 * np.pi * x  
        r     = np.sqrt(y)
        return theta, r
    def area_preserve_circ2sphere(theta, r):  
        phi = 2 * np.arcsin(r)
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        return x, y, z
        
    x, y     = gen_fib_lattice(num_points)
    theta, r = area_preserve_rec2circ(x, y)
    x, y, z  = area_preserve_circ2sphere(theta, r)
    
    return np.column_stack([x, y, z])

def find_neighbor_on_sphere_fibonacci(vertices, fibonacci_indices):
    def find_pairs_with_shift_sorted(arr, shift):
        arr     = np.sort(np.asarray(arr))
        shifted = arr + shift
        pos     = np.searchsorted(arr, shifted)
        pos     = np.clip(pos, 0, len(arr) - 1)
        mask    = arr[pos] == shifted
        original_indices = np.where(mask)[0]
        target_indices   = pos[mask]
        return original_indices, target_indices
    def find_nearest_neighbors(vertices, target_index, k=6):
        target = vertices[target_index]
        deltas = vertices - target
        sq_distances = np.einsum('ij,ij->i', deltas, deltas)
        sq_distances[target_index] = np.inf
        sorted_indices = np.argsort(sq_distances)
        return sorted_indices[:k]
    def find_fibonacci_neighbor_interval(vertices, fibonacci_indices, center_point, k=4):
        center_point_neighbor = find_nearest_neighbors(vertices, center_point, 4)
        interval = np.unique(np.abs([fibonacci_indices[center_point] - fibonacci_indices[center_point_neighbor[i]] for i in range(k)]))
        return interval
    def find_neighbor_by_interval(fibonacci_indices, fibonacci_interval):
        source_indices, target_indices = [], []
        for shift in fibonacci_interval:
            source_index, target_index = find_pairs_with_shift_sorted(fibonacci_indices, shift)
            source_indices.append(source_index)
            target_indices.append(target_index)
        source_indices = np.concatenate(source_indices)
        target_indices = np.concatenate(target_indices)
        return np.unique(np.sort(np.array([source_indices, target_indices]), axis=0), axis=1).T
        
    center_point       = find_nearest_neighbors(np.vstack([vertices, [0.5, 0.5, 0.5]]), -1, 1)
    fibonacci_interval = find_fibonacci_neighbor_interval(vertices, fibonacci_indices, center_point)
    edges              = find_neighbor_by_interval(fibonacci_indices, fibonacci_interval)
    
    return edges