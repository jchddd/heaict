from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod
import numpy as np


class SurrogateModel(ABC):
    '''
    Base module for a surrogate model

    Methods:
        - fit (abstractmethod): Fit models for each object by normalizated X and Y
        - evaluate (abstractmethod): Evaluate mean value, gradient, and Hessian of the objective by normalized X
        - evaluate_constr: Evaluate constraints (G<=0) by normalized X
        - evaluate_feasibility: Evaluate the feasibility by normalized X
    Attributes:
        - real_problem (RealProblem): the real problem
        - normalization (Normalization): class used for normalizing X and Y
        - xl ((n_var, ) np.array): lower boundaries for normalized X
        - xu ((n_var, ) np.array): upper boundaries for normalized X        
    '''
    def __init__(self, real_problem):
        '''
        Parameter:
            - real_problem (mobo.Problem.Real_Problem)
        '''
        self.real_problem = real_problem
        
        self.normalization = Normalization()
        self.normalization.fit(bounds=np.array([real_problem.xl, real_problem.xu]))

        self.xl = self.normalization.do(X=real_problem.xl)
        self.xu = self.normalization.do(X=real_problem.xu)

    @abstractmethod
    def fit(self, X, Y, update=False):
        pass
        
    @abstractmethod
    def evaluate(self, X, return_values_of=['F', 'dF', 'hF']):
        pass

    def evaluate_constr(self, X):
        '''
        Evaluate constraints (G<=0) by normalized X

        Parameter:
            - X ((num_samples, n_var) np.array): the normalized X
        Return:
            - G ((num_samples, n_constr) np.array) if num_samples > 1
            - G ((n_constr) np.array) if num_samples == 1
            - None if num_samples == 0
        '''
        if self.real_problem.n_constr > 0:
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            X = self.normalization.undo(X)
            G = self.real_problem.evaluate_constraint_batch(X)
            G = G[0] if X.shape[0] == 1 else G
        else:
            G = None
        return G

    def evaluate_feasibility(self, X):
        '''
        Evaluate the feasibility by normalized X

        Parameter:
            - X ((num_samples, n_var) np.array): the normalized X
        Return:
            - feasibility for each sample ((num_samples, ) bool type np.array)
        '''
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        X = self.normalization.undo(X)
        
        if self.real_problem.n_constr == 0:
            feasibility_vector = np.full(len(X), True)
        else:
            G = self.real_problem.evaluate_constraint_batch(X)
            feasibility_vector = np.all(G <= 0, axis=1)
        return feasibility_vector


class Normalization():
    def __init__(self):
        self.x_scaler = BoundedScaler()
        self.y_scaler = StandardScaler()

    def fit(self, bounds=None, Y=None):
        if bounds is not None:
            self.x_scaler.fit(bounds)
        if Y is not None:
            self.y_scaler.fit(Y)
    
    def do(self, X=None, Y=None):
        if X is not None and Y is not None:
            return self.x_scaler.transform(X), self.y_scaler.transform(Y)
        elif X is not None:
            return self.x_scaler.transform(X)
        elif Y is not None:
            return self.y_scaler.transform(Y)

    def undo(self, X=None, Y=None):
        if X is not None and Y is not None:
            return self.x_scaler.inverse_transform(X), self.y_scaler.inverse_transform(Y)
        elif X is not None:
            return self.x_scaler.inverse_transform(X)
        elif Y is not None:
            return self.y_scaler.inverse_transform(Y)


class BoundedScaler():
    def __init__(self):
        self.bounds = None
        
    def fit(self, bounds):
        self.bounds = bounds
        return self

    def transform(self, X):
        return np.clip((X - self.bounds[0]) / (self.bounds[1] - self.bounds[0]), 0, 1)

    def inverse_transform(self, X):
        return np.clip(X, 0, 1) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]