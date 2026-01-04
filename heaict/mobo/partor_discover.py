from heaict.mobo.utility import print_with_timestamp, find_pareto_front
from heaict.mobo.buffer  import Buffer

from scipy.optimize import minimize, NonlinearConstraint
from scipy.optimize import minimize
from scipy.linalg   import null_space
from pymoo.indicators.hv import HV
import numpy as np


class ParetoDiscovery():
    '''
    The Pareto discovery algorithm introduced by: Schulz, Adriana, et al. "Interactive exploration of design trade-offs." ACM Transactions on Graphics (TOG) 37.4 (2018): 1-14.

    Method:
        - solve: Use Pareto discovery to suggest points on the Pareto front for the next batch
    '''
    def __init__(self, 
                 n_gen=6, 
                 pop_size=30, 
                 batch_size=10,
                 n_grid_sample=1000,
                 perturb_method='origin',
                 delta_p=10,
                 delta_s=1,
                 delta_b=0.2,
                 cell_num=100,
                 cell_size=10,
                 cell_shape='triangle',
                 sparse_approx=True):
        '''
        Parameters:
            - n_gen (int): number of generations (number of iterations in the pareto front search process). Default = 6
            - pop_size (int): population size (Initial sample quantity). Default = 30
            - batch_size (int): the number of candidate samples collected in pareto discovery each iteration. Default = 10
            - n_grid_sample (int): number of collected samples on local Pareto manifold. Default = 1000
            - perturb_method (int): method to perturb samples. 'origin' or 'gaussian'. Default = 'origin'
            - delta_p (float): factor of perturbation in stochastic sampling. Default = 10
            - delta_s (float): scaling factor for choosing reference point in local optimization. Default = 1
            - delta_b (float): unary energy normalization constant for sparse approximation. Default = 0.2
            - cell_num (int): number of cells in performance buffer. Default = 100
            - cell_size (int): maximum number of samples inside each cell of the performance buffer. Default = 10
            - cell_shape (str): type of 3D buffer cell vertices, 'triangle' or 'sphere'. Default = 'triangle'
            - sparse_approx (bool): perform sparse approximation for suggesting the next batch sample, needs pygco. Default = False
        '''
        # 
        self.n_gen = n_gen
        self.pop_size = pop_size
        self.batch_size = batch_size
        self.delta_s = delta_s
        self.n_grid_sample = n_grid_sample
        # perturb related
        self.perturb_method = perturb_method
        self.delta_p = delta_p
        # buffer related
        self.delta_b = delta_b
        self.cell_num = cell_num
        self.cell_size = cell_size
        self.cell_shape = cell_shape
        self.sparse_approx = sparse_approx
        # whether save history
        self.save_samples = False
        self.sample_history = {'X': [], 'Y': []}

    def solve(self, X, surrogate_model, save_samples=False):
        '''
        Use Pareto discovery to suggest points on the Pareto front for the next batch

        Parameters:
            - X ((n, n_var) np.array): X without normalization
            - surrogate_model (SurrogateModel)
        Return:
            - Recommended samples without normalization for next batch (batch_size, n_var) np.array
        '''
        self.surrogate_model = surrogate_model
        X = self.surrogate_model.normalization.do(X)
        print_with_timestamp(f'--- ParetoDiscovery')
        self._initialize(X)
        for _ in range(self.n_gen):
            print(f'generation {str(_ + 1)}: processing', end='')
            self._next()
            print_with_timestamp(f'\rgeneration {str(_ + 1)}: completed')
        print(f'Suggest next batch: processing', end='')
        self._finalize()
        print_with_timestamp(f'\rSuggest next batch: completed')
        
        return self.surrogate_model.normalization.undo(self.X_next)

    def _initialize(self, X):
        '''
        initialize buffer cells and get first population from them
        '''
        self.patch_id = 0
        self.buffer = Buffer(self.surrogate_model.real_problem.n_obj, self.cell_num, self.cell_size, self.delta_b, self.cell_shape)
        Y = self.surrogate_model.evaluate(X, return_values_of=['F'])
        patch_ids = np.full(self.pop_size, self.patch_id)
        self.buffer.insert(X, Y, patch_ids)
        self.patch_id += 1
        
        self.pop = {'X': self.buffer.sample(self.pop_size), 'X_init': X, 'Y_init': Y}

        if self.save_samples:
            self.sample_history['X'].append(self.pop['X'])
            self.sample_history['Y'].append(self.surrogate_model.evaluate(self.pop['X'], return_values_of=['F']))
        
    def _next(self):
        new_samples, patch_ids = self._pareto_discover()
        new_Y = self.surrogate_model.evaluate(new_samples, return_values_of=['F'])
        self.buffer.insert(new_samples, new_Y, np.array(patch_ids) + self.patch_id)
        self.patch_id += self.pop_size
        self.pop['X'] = self.buffer.sample(self.pop_size)

        if self.save_samples:
            self.sample_history['X'].append(self.pop['X'])
            self.sample_history['Y'].append(self.surrogate_model.evaluate(self.pop['X'], return_values_of=['F']))

    def _pareto_discover(self):
        xs = self._stochastic_sampling()
        eval_func = self.surrogate_model.evaluate
        constr_func = self.surrogate_model.evaluate_constr
        n_constr = self.surrogate_model.real_problem.n_constr
        bounds = [self.surrogate_model.xl, self.surrogate_model.xu]
        origin = self.buffer.origin
        delta_s = self.delta_s
        n_grid_sample = self.n_grid_sample
        
        ys = eval_func(xs, return_values_of=['F'])
        if self.save_samples:
            self.sample_history['X'].append(xs)
            self.sample_history['Y'].append(ys)
        new_origin = np.minimum(origin, np.min(ys, axis=0))
        if (new_origin != origin).any():
            new_origin -= 1e-2
        fs = ys - new_origin
    
        x_samples_all = []
        patch_ids = []
        for i, (x, y, f) in enumerate(zip(xs, ys, fs)):
            x_opt = _local_optimization(x, y, f, eval_func, constr_func, n_constr, bounds, delta_s)
            directions = _get_optimization_directions(x_opt, eval_func, bounds, constr_func)
            x_samples = _first_order_approximation(x_opt, constr_func, n_constr, directions, bounds, n_grid_sample)
            x_samples_all.append(x_samples)
            patch_ids.extend([i] * len(x_samples))        
    
        return np.vstack(x_samples_all), patch_ids
        

    def _stochastic_sampling(self):
        current_population = self.pop.get('X').copy()
        num_target         = self.pop_size
        stochastic_samples = np.zeros((0, current_population.shape[1]), current_population.dtype)
        while stochastic_samples.shape[0] < num_target:
            if   self.perturb_method == 'origin':
                stochastic_direction = np.random.random(current_population.shape)
                stochastic_direction /= np.expand_dims(np.linalg.norm(stochastic_direction, axis=1), axis=1)
                stochastic_scaling   = np.random.uniform(0.2, 1.0) * self.delta_p # np.random.random()
                perturb_population   = current_population + 1.0 / (2 ** stochastic_scaling) * stochastic_direction # * current_population
            elif self.perturb_method == 'gaussian':
                gaussian_noise     = np.random.normal(loc=0.0, scale=self.delta_p, size=current_population.shape)
                perturb_population = current_population + gaussian_noise
            perturb_population   = np.clip(perturb_population, self.surrogate_model.xl, self.surrogate_model.xu)
            perturb_feasibility  = self.surrogate_model.evaluate_feasibility(perturb_population)
            if np.any(perturb_feasibility): stochastic_samples = np.vstack([stochastic_samples, perturb_population[perturb_feasibility]])
        return stochastic_samples[:num_target]

    def _finalize(self):
        X_all_explore, Y_all_explore, _ = self.buffer.flattened()
        
        # Confirm the candidate samples
        select_from_all = True
        perform_approx = False
        if self.sparse_approx:
            self.approx_labels, self.approx_xs, self.approx_ys = self.buffer.sparse_approximation()
            if self.approx_xs.shape[0] > self.batch_size:
                select_from_all = False
                perform_approx = True
                X_candidate = self.approx_xs
                Y_candidate = self.approx_ys
                idx_choices = np.ma.array(np.arange(X_candidate.shape[0]), mask=False)
                idx_choices_without_label = np.ma.array(np.arange(X_candidate.shape[0]), mask=False)
        else:
            self.Knearest_X, self.Knearest_Y = self.buffer.k_nearest_sparse(1)
            if self.Knearest_X.shape[0] > self.batch_size:
                select_from_all = False
                X_candidate = self.Knearest_X
                Y_candidate = self.Knearest_Y
                idx_choices = np.ma.array(np.arange(X_candidate.shape[0]), mask=False)
        if select_from_all:
            X_candidate = X_all_explore
            Y_candidate = Y_all_explore
            idx_choices = np.ma.array(np.arange(X_candidate.shape[0]), mask=False)

        # The reference points for calculating hypervolume and lists for storing results
        curr_pfront, _ = find_pareto_front(self.pop['Y_init'])
        reference_point = np.max(np.vstack([Y_all_explore, self.pop['Y_init']]), axis=0)
        hv_calculator = HV(ref_point=reference_point)
        next_batch_indices = []
        family_lbls_next = []

        # select batch_size number of samples by maximize hypervolume
        for _ in range(self.batch_size):
            if not select_from_all and len(idx_choices.compressed())==0:
                idx_choices = idx_choices_without_label.copy()
            curr_hv = hv_calculator.do(curr_pfront)
            max_hv_contrib = 0.
            max_hv_idx = -1
            for idx in idx_choices.compressed():
                new_hv = hv_calculator.do(np.vstack([curr_pfront, Y_candidate[idx]]))
                hv_contrib = new_hv - curr_hv
                if hv_contrib > max_hv_contrib:
                    max_hv_contrib = hv_contrib
                    max_hv_idx = idx
            if max_hv_idx == -1:
                max_hv_idx = np.random.choice(idx_choices.compressed())
                
            idx_choices.mask[max_hv_idx] = True
            curr_pfront = np.vstack([curr_pfront, Y_candidate[max_hv_idx]])
            next_batch_indices.append(max_hv_idx)
            family_lbls_next = [-1 if not perform_approx else self.approx_labels[max_hv_idx]]
            if not select_from_all and self.sparse_approx:
                idx_choices_without_label.mask[max_hv_idx] = True
                family_ids = np.where(perform_approx == self.approx_labels[max_hv_idx])[0]
                for fid in family_ids:
                    idx_choices.mask[fid] = True

        # store candidate
        self.X_next = X_candidate[next_batch_indices].copy()
        self.Y_next = Y_candidate[next_batch_indices].copy()
        self.L_next = family_lbls_next

        # find current pareto front
        pfront, idx_pfront = find_pareto_front(Y_all_explore)
        self.pfront = pfront
        self.pfrset = X_all_explore[idx_pfront]


def _local_optimization(x, y, f, eval_func, constr_func, n_constr, bounds, delta_s):
    '''
    Local optimization of generated stochastic samples by minimizing distance to the target
    '''
    # choose reference point z
    f_norm = np.linalg.norm(f)
    s = 2.0 * f / np.sum(f) - 1 - f / f_norm
    s /= np.linalg.norm(s)
    z = y + s * delta_s * np.linalg.norm(f)
    # optimization objective
    def fun(x):
        fx = eval_func(x, return_values_of=['F'])
        return np.linalg.norm(fx - z)
    # constraint function
    if n_constr > 0:
        nlc = NonlinearConstraint(constr_func, [-np.inf] * n_constr, [0] * n_constr)
    # jacobian of the objective
    if   n_constr == 0:
        def jac(x):
            fx, dfx = eval_func(x, return_values_of=['F', 'dF'])
            grad = ((fx - z) / np.linalg.norm(fx - z)) @ dfx
            return grad
    elif n_constr > 0:
        def jac_constr(x):
            fx, dfx = eval_func(x, return_values_of=['F', 'dF'])
            grad = ((fx - z) / np.linalg.norm(fx - z)) @ dfx
            g = constr_func(x)
            if np.any(g > 0):
                return -grad
            else:
                return grad
    # do optimization
    if   n_constr == 0:
        res = minimize(fun, x, method='L-BFGS-B', jac=jac,        bounds=np.array(bounds).T)
    elif n_constr  > 0:
        res = minimize(fun, x, method='SLSQP',    jac=jac_constr, bounds=np.array(bounds).T, constraints=nlc, tol=1e-4)
    # return result
    x_opt = res.x
    if res.success == True:
        return x_opt
    else:
        return x


def _get_optimization_directions(x_opt, eval_func, bounds, constr_func):
    '''
    Getting the directions to explore local pareto manifold.
    '''
    F, DF, HF = eval_func(x_opt, return_values_of=['F', 'dF', 'hF'])
    
    G, DG, HG = _get_box_const_value_jacobian_hessian(x_opt, bounds, constr_func)
    
    alpha, beta = _get_kkt_dual_variables(F, G, DF, DG)

    n_obj, n_var, n_active_const = len(F), len(x_opt), len(G) if G is not None else 0
    if n_active_const > 0:
        H = HF.T @ alpha + HG.T @ beta
    else:
        H = HF.T @ alpha
    alpha_const = np.concatenate([np.ones(n_obj), np.zeros(n_active_const + n_var)])
    if n_active_const > 0:
        comp_slack_const = np.column_stack([np.zeros((n_active_const, n_obj + n_active_const)), DG])
        DxHx = np.vstack([alpha_const, comp_slack_const, np.column_stack([DF.T, DG.T, H])])
    else:
        DxHx = np.vstack([alpha_const, np.column_stack([DF.T, H])])
    directions = null_space(DxHx)

    eps = 1e-8
    directions[np.abs(directions) < eps] = 0.0
    return directions
    
    
def _get_box_const_value_jacobian_hessian(x, bounds, constr_func):
    '''
    Getting the value, jacobian and hessian of active box constraints.
    '''
    active_idx, upper_active_idx, _ = _get_active_box_const(x, bounds)
    n_active_const, n_var = len(active_idx), len(x)

    g = constr_func(x.reshape(1, -1))
    # if np.any(g[2:] > 0):
        # n_active_const += 1

    if n_active_const > 0:
        G = np.zeros(n_active_const)
        DG = np.zeros((n_active_const, n_var))
        for i, idx in enumerate(active_idx):
            constraint = np.zeros(n_var)
            if idx in upper_active_idx: 
                constraint[idx] = 1
            else:
                constraint[idx] = -1
            DG[i] = constraint
        # if np.any(g[2:] > 0):
            # constraint = np.array([-1, -1, -1, 1])
            # for idx in active_idx:
                # constraint[idx] = 0
            # DG[-1] = constraint
        HG = np.zeros((n_active_const, n_var, n_var))
        return G, DG, HG
    else:
        return None, None, None


def _get_active_box_const(x, bounds):
    '''
    Getting the indices of active box constraints.
    '''
    eps = 1e-8
    upper_active = bounds[1] - x < eps
    lower_active = x - bounds[0] < eps
    active = np.logical_or(upper_active, lower_active)
    active_idx, upper_active_idx, lower_active_idx = np.where(active)[0], np.where(upper_active)[0], np.where(lower_active)[0]
    return active_idx, upper_active_idx, lower_active_idx


def _get_kkt_dual_variables(F, G, DF, DG):
    '''
    Optimizing for dual variables alpha and beta in KKT conditions.
    '''
    n_obj = len(F)
    n_active_const = len(G) if G is not None else 0
    if n_active_const > 0:

        def fun(x, n_obj=n_obj, DF=DF, DG=DG):
            alpha, beta = x[:n_obj], x[n_obj:]
            objective = alpha @ DF + beta @ DG
            return 0.5 * objective @ objective

        def jac(x, n_obj=n_obj, DF=DF, DG=DG):
            alpha, beta = x[:n_obj], x[n_obj:]
            objective = alpha @ DF + beta @ DG
            return np.vstack([DF, DG]) @ objective

        const = {'type': 'eq', 
            'fun': lambda x, n_obj=n_obj: np.sum(x[:n_obj]) - 1.0, 
            'jac': lambda x, n_obj=n_obj: np.concatenate([np.ones(n_obj), np.zeros_like(x[n_obj:])])}
    
    else:
        
        def fun(x, DF=DF):
            objective = x @ DF
            return 0.5 * objective @ objective

        def jac(x, DF=DF):
            objective = x @ DF
            return DF @ objective

        const = {'type': 'eq', 
                'fun': lambda x: np.sum(x) - 1.0, 
                'jac': np.ones_like}

    bounds = np.array([[0.0, np.inf]] * (n_obj + n_active_const))
    
    alpha_init = np.random.random(len(F))
    alpha_init /= np.sum(alpha_init)
    beta_init = np.zeros(n_active_const)
    x_init = np.concatenate([alpha_init, beta_init])
    res = minimize(fun, x_init, method='SLSQP', jac=jac, bounds=bounds, constraints=const)
    x_opt = res.x
    alpha_opt, beta_opt = x_opt[:n_obj], x_opt[n_obj:]
    return alpha_opt, beta_opt


def _first_order_approximation(x_opt, constr_func, n_constr, directions, bounds, n_grid_sample):
    '''
    Exploring new samples from local manifold (first order approximation of pareto front).
    '''
    n_var = len(x_opt)
    lower_bound, upper_bound = bounds[0], bounds[1]
    active_idx, _, _ = _get_active_box_const(x_opt, bounds)
    n_active_const = len(active_idx)
    n_obj = len(directions) - n_var - n_active_const

    x_samples = np.array([x_opt])
    d_alpha, d_beta, d_x = directions[:n_obj], directions[n_obj:n_obj + n_active_const], directions[-n_var:]
    eps = 1e-8
    if np.linalg.norm(d_x) < eps:
        return x_samples
    direction_dim = d_x.shape[1]

    if direction_dim > n_obj - 1:
        indices = np.random.choice(np.arange(direction_dim), n_obj -  1)
        while np.linalg.norm(d_x[:, indices]) < eps:
            indices = np.random.choice(np.arange(direction_dim), n_obj - 1)
        d_x = d_x[:, indices]
    elif direction_dim < n_obj - 1:
        return x_samples
    
    d_x /= np.linalg.norm(d_x)
    bound_scale = np.expand_dims(upper_bound - lower_bound, axis=1)
    d_x *= bound_scale
    loop_count = 0
    while len(x_samples) < n_grid_sample:
        curr_dx_samples = np.sum(np.expand_dims(d_x, axis=0) * np.random.random((n_grid_sample, 1, n_obj - 1)), axis=-1)
        curr_x_samples = np.expand_dims(x_opt, axis=0) + curr_dx_samples
        flags = np.logical_and((curr_x_samples <= upper_bound).all(axis=1), (curr_x_samples >= lower_bound).all(axis=1))
        if   n_constr == 1:
            flags = np.logical_and(flags, constr_func(curr_x_samples) <= 0)
        elif n_constr  > 1:
            flags = np.logical_and(flags, np.all(constr_func(curr_x_samples) <= 0, axis=1))
        valid_idx = np.where(flags)[0]
        x_samples = np.vstack([x_samples, curr_x_samples[valid_idx]])
        loop_count += 1
        if loop_count > 10:
            break
    x_samples = x_samples[:n_grid_sample]
    return x_samples
