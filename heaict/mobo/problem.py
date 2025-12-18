from abc import ABC, abstractmethod
import numpy as np


class Real_Problem(ABC):
    '''
    A class that used to define the real problem

    Methods:
        - evaluate_objective_elementwise: Evaluate the objective value for a single sample.
        - evaluate_objective_batch (abstractmethod): Evaluate the objective value for a batch of samples.
        - evaluate_constraint_elementwise: Evaluate the constraint (G <= 0) for a single sample.
        - evaluate_constraint_batch (abstractmethod): Evaluate the constraint (G <= 0) for a batch of samples.
        - calc_pareto_front: Calculate the true Pareto front
    Attributes:
        - n_var (int): number of variables.
        - n_obj (int): number of objectives.
        - n_constr (int): number of constraints.
        - var_lb ((n_var,) list or float): lower boundaries for variables.
        - var_ub ((n_var,) list or float): upper boundaries for variables.
    '''
    def __init__(self, n_var, n_obj, n_constr, var_lb, var_ub):
        '''
        Parameters:
            - n_var (int): number of variables.
            - n_obj (int): number of objectives.
            - n_constr (int): number of constraints.
            - var_lb (float or (n_var,) list): lower boundaries for variables.
            - var_ub (float or (n_var,) list): upper boundaries for variables.
        '''
        self.n_var = n_var
        self.n_obj = n_obj
        self.n_constr = n_constr
        self.xl = np.array(var_lb if type(var_lb) is list else [var_lb] * n_var)
        self.xu = np.array(var_ub if type(var_ub) is list else [var_ub] * n_var)

    def evaluate_objective_elementwise(self, x):
        '''
        Evaluate the objective value for a single sample

        Parameter:
            - x ((n_var,) np.array)
        Return:
            - objective values as (n_obj,) np.array
        '''
        pass

    @abstractmethod
    def evaluate_objective_batch(self, X):
        '''
        Evaluate the objective value for a batch of samples

        Parameter:
            - X ((num_samples, n_var) np.array)
        Return:
            - objective values as (num_samples, n_obj) np.array
        '''
        pass
        
    def evaluate_constraint_elementwise(self, x):
        '''
        Evaluate the constraint (G <= 0) for a single sample

        Parameter:
            - x ((n_var,) np.array)
        Return:
            - constraint values as (n_constr,) np.array
        '''
        pass
        
    @abstractmethod
    def evaluate_constraint_batch(self, X):
        '''
        Evaluate the constraint (G <= 0) for a batch of samples

        Parameter:
            - X (num_samples, n_var) np.array
        Return:
            - constraint values as (num_samples, n_constr) np.array if n_constr > 0 else None
        '''
        pass
        
    def calc_pareto_front(self):
        '''
        Calculate the true Pareto front

        Return:
            - a (num_pareto_points, n_obj) np.array
        '''
        pass
        
        
class ZDT1(Real_Problem):
    def __init__(self, n_var=10):
        super().__init__(n_var, 2, 0, 0, 1)

    def evaluate_objective_batch(self, X):
        f1 = X[:, 0]
        g = 1 + 9.0 / (self.n_var - 1) * np.sum(X[:, 1:], axis=1)
        f2 = g * (1 - np.power((f1 / g), 0.5))

        return np.column_stack([f1, f2])

    def evaluate_constraint_batch(self, X):
        return None
        
    def calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0, 1, n_pareto_points)
        return np.array([x, 1 - np.sqrt(x)]).T


class ZDT2(Real_Problem):
    def __init__(self, n_var=10):
        super().__init__(n_var, 2, 0, 0, 1)

    def evaluate_objective_batch(self, X):
        f1 = X[:, 0]
        c = np.sum(X[:, 1:], axis=1)
        g = 1.0 + 9.0 * c / (self.n_var - 1)
        f2 = g * (1 - np.power((f1 * 1.0 / g), 2))

        return np.column_stack([f1, f2])

    def evaluate_constraint_batch(self, X):
        return None
        
    def calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0, 1, n_pareto_points)
        return np.array([x, 1 - np.power(x, 2)]).T


class ZDT3(Real_Problem):
    def __init__(self, n_var=10):
        super().__init__(n_var, 2, 0, 0, 1)

    def evaluate_objective_batch(self, X):
        f1 = X[:, 0]
        c = np.sum(X[:, 1:], axis=1)
        g = 1.0 + 9.0 * c / (self.n_var - 1)
        f2 = g * (1 - np.power(f1 * 1.0 / g, 0.5) - (f1 * 1.0 / g) * np.sin(10 * np.pi * f1))

        return np.column_stack([f1, f2])

    def evaluate_constraint_batch(self, X):
        return None
        
    def calc_pareto_front(self, n_pareto_points=100):
        regions = [[0, 0.0830015349],
                   [0.182228780, 0.2577623634],
                   [0.4093136748, 0.4538821041],
                   [0.6183967944, 0.6525117038],
                   [0.8233317983, 0.8518328654]]

        pf = []

        for r in regions:
            x1 = np.linspace(r[0], r[1], int(n_pareto_points / len(regions)))
            x2 = 1 - np.sqrt(x1) - x1 * np.sin(10 * np.pi * x1)
            pf.append(np.array([x1, x2]).T)
            
        pf = np.row_stack(pf)
        return pf