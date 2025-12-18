from heaict.mobo.utility import print_with_timestamp, generate_random_initial_samples

from pymoo.config import Config
import numpy as np


Config.warnings['not_compiled'] = False


class MOBO():
    '''
    Basic class for perform Multi-objective Bayesian optimization

    Methods:
        - optimize: perform optimization
    Attribures:
        - surrogate_model: surrogate model for rapidly predicting objective values
        - algorithm: algorithm to solve Multi-objective problem
    '''
    def __init__(self, surrogate_model, algorithm):
        '''
        Parameters:
            - surrogate_model: surrogate model for rapidly predicting objective values
            - algorithm: algorithm to solve Multi-objective problem
        '''
        self.surrogate_model = surrogate_model
        self.algorithm = algorithm

    def optimize(self, n_init, n_iter, X_init=None, Y_init=None, update_final=False, return_real=False):
        '''
        Perform optimization

        Parameters:
            - n_init (int): number of initial samples
            - n_iter (int): number of iterations
            - X_init ((n_init, n_var) np.array or None): initial X samples. if None general randomly. Default = None
            - Y_init ((n_init, n_obj) np.array or None): initial Y values. if None, predict by Real Problem. Default = None
            - update_final (bool): using samples on the last iteration to update surrogate model. Default = False
            - return_real (bool): return values evaluating by real problem. if not, by surrogate model. Default = False
        Returns:
            - X and Y data with a number of n_init + n_iter * batch_size
        '''
        print_with_timestamp(f'------ Start optimization')
        # get init samples
        if X_init is None:
            X = generate_random_initial_samples(self.surrogate_model.real_problem, n_init)
            print(f'Generate {n_init} initial samples')
        else:
            X = X_init
            print(f'Read in {X_init.shape[0]} initial samples')
        # loop for n_iter
        for _ in range(n_iter):
            print_with_timestamp(f'------ Iteration {str(_ + 1)}')
            # update surrogate model
            if _ == 0:
                if Y_init is None:
                    self.surrogate_model.fit(X)
                else:
                    self.surrogate_model.fit(X, Y_init)
            else:
                self.surrogate_model.fit(X_next, update=True)
            # solve next X
            X_next = self.algorithm.solve(X, self.surrogate_model)
            X = np.vstack([X, X_next])
            print(f'Add {str(X_next.shape[0])} to training set')
        print(f'------ End optimization')
        # update final X_next
        if update_final:
            print(f'Update final samples into surrogate model')
            self.surrogate_model.fit(X_next, update=True)
        # return results
        if return_real:
            return self.surrogate_model.X_store, self.surrogate_model.Y_store
        else:
            X_nor = self.surrogate_model.normalization.do(X=X)
            Y = self.surrogate_model.normalization.undo(Y=self.surrogate_model.evaluate(X_nor, return_values_of=['F']))
            return X, Y