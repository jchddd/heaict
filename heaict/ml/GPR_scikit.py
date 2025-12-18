from heaict.ml.SM_main import SurrogateModel
from heaict.ml.utility import get_z_value, safe_divide, print_with_timestamp

from sklearn.metrics                  import mean_squared_error, mean_absolute_error, r2_score
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
from sklearn.gaussian_process         import GaussianProcessRegressor
from sklearn.exceptions               import ConvergenceWarning

from scipy.spatial.distance import cdist
from scipy.optimize         import minimize
import numpy as np
import warnings


warnings.filterwarnings("ignore", category=ConvergenceWarning)


class GPR(SurrogateModel):
    '''
    Gaussian process regression surrogate model implemented based on scikit-learn

    Methods:
        - fit: Fit Gaussian Processes for each object by raw X samples
        - evaluate: Evaluate mean value, gradient, and Hessian of the objective by normalized X
        - print_summary: Prints a summary of the parameters and variables contained in GPs
        - print_metric: Prints a summary of metrics for GPs
    Attributes:
        - gps ((n_obj, ) list): list containing GP models for each objective
        - gp_kernel ((n_obj, ) list): specify kernel type for each GP model
        - set_kernel (bool): whether kernel is specified
        - main_kernel_bounds ((2, ) tuple): bounds for main kernel length scale
        - scale_kernel_bounds ((2, ) tuple): bounds for scale kernel variance
        - bias_kernel_bounds ((2, ) tuple): bounds for bias kernel variance
    '''
    def __init__(self, 
                 real_problem, gp_kernel='RBF', 
                 main_kernel_bounds=(np.sqrt(1e-3), np.sqrt(1e3)),
                 scale_kernel_bounds=(np.sqrt(1e-3), np.sqrt(1e3)),
                 bias_kernel_bounds=(np.exp(-6), np.exp(0))
                ):
        '''
        Parameters:
            - real_problem (RealProblem): the real problem
            - gp_kernel (str, (n_obj, ) list or a [0, 1] float): set kernels for each GP. Default = 'RBF'
              The available kernel functions now are: 'RBF', 'Matern12', 'Matern32', 'Matern52'
              If str, all GPs will use this type of kernel. If list, choose kernels for each GP by hand.
              If float (alpha), will calculate (1-alpha) * AIC + alpha * BIC score for each kernel, and then
              select the best one for each GP automatically. A larger amount of data is in situ with a higher alpha.
            - main_kernel_bounds ((2, ) tuple): bounds for main kernel length scale
            - scale_kernel_bounds ((2, ) tuple): bounds for scale kernel variance
            - bias_kernel_bounds ((2, ) tuple): bounds for bias kernel variance
        '''
        super().__init__(real_problem)

        self.gps = []
        self.gp_kernel = gp_kernel
        self.set_kernel = False

        self.main_kernel_bounds = main_kernel_bounds
        self.scale_kernel_bounds = scale_kernel_bounds
        self.bias_kernel_bounds = bias_kernel_bounds

        self.X_store = None
        self.Y_store = None

    def fit(self, X, Y=None, update=False):
        '''
        Fit Gaussian Processes for each object by raw X samples

        Parameter:
            - X ((num_samples, n_var) np.array): the raw X data that is used to fit Gaussian Processes
            - Y (None or (num_samples, n_obj) np.array): the raw Y data, if None, use real problem to evaluate
            - update (bool): whether the input X is the updated data, if so, combine stored and input X data. Default = False
        '''
        if Y is None:
            print_with_timestamp(f'Evaluate {len(X)} data points from real problem')
            Y = self.real_problem.evaluate_objective_batch(X)
        if update and self.X_store is not None:
            X = np.vstack([self.X_store, X])
            Y = np.vstack([self.Y_store, Y])
        print_with_timestamp(f'Fit surrogate model with {len(X)} data points')
        self.normalization.fit(Y=Y)
        X_nor, Y_nor = self.normalization.do(X, Y)
        
        self._set_surrogate_model(X_nor, Y_nor)
        for i, gp in enumerate(self.gps):
            gp.fit(X_nor, Y_nor[:, i])

        print_with_timestamp(f'Surrogate models constructed successfully')

        self.X_store = X
        self.Y_store = Y
        self.print_metric()

    @staticmethod
    def constrained_optimization(obj_func, initial_theta, bounds):
        opt_res = minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds)
        return opt_res.x, opt_res.fun

    def _set_surrogate_model(self, X_nor, Y_nor):
        self.gps = []
        
        select_kernel = False
        if not self.set_kernel:
            self.set_kernel = True
            if   type(self.gp_kernel) is list:
                self.gp_kernel = self.gp_kernel
            elif type(self.gp_kernel) is str:
                self.gp_kernel = [self.gp_kernel] * self.real_problem.n_obj
            else:
                alpha = self.gp_kernel
                self.gp_kernel = []
                select_kernel = True
                print(f"Selected kernels automatically through AIC and BIC scores:")
            
        for i in range(self.real_problem.n_obj):
            kernels = {}
            kernels['RBF'] = RBF(length_scale=np.ones(self.real_problem.n_var), length_scale_bounds=self.main_kernel_bounds)
            kernels['Matern12'] = Matern(length_scale=np.ones(self.real_problem.n_var), length_scale_bounds=self.main_kernel_bounds, nu=0.5 * 1)
            kernels['Matern32'] = Matern(length_scale=np.ones(self.real_problem.n_var), length_scale_bounds=self.main_kernel_bounds, nu=0.5 * 3)
            kernels['Matern52'] = Matern(length_scale=np.ones(self.real_problem.n_var), length_scale_bounds=self.main_kernel_bounds, nu=0.5 * 5)
            kernels = {kernel: ConstantKernel(constant_value=1.0, constant_value_bounds=self.scale_kernel_bounds) * main_kernel + \
                      ConstantKernel(constant_value=1e-2, constant_value_bounds=self.bias_kernel_bounds) for kernel, main_kernel in kernels.items()}

            if select_kernel:
                print(f'Select kernel for obj {str(i)}')
                selected_kernel = self._select_kernel_by_abic(alpha, (X_nor, Y_nor[:, i: i + 1]), kernels)
                self.gp_kernel.append(selected_kernel)
            
            gp = GaussianProcessRegressor(kernel=kernels[self.gp_kernel[i]], optimizer=self.constrained_optimization) # , alpha=np.var(Y_nor[:, i: i + 1])*0.01
            self.gps.append(gp)
            
        print(f"initialize surrogate models with kernel: {' '.join(self.gp_kernel)}")

    def _select_kernel_by_abic(self, alpha, data, kernels):
        best_abic = np.inf
        best_kernel = None
        
        for kernel_name, kernel in kernels.items():
            model = GaussianProcessRegressor(kernel=kernel, optimizer=self.constrained_optimization)
            model.fit(data[0], data[1])
            
            nll = -1 * model.log_marginal_likelihood()
            num_params = len(model.kernel.theta) + 1
            n = len(data[0])
            abic = (1 - alpha) * (2 * num_params + 2 * nll) + alpha * (np.log(n) * num_params + 2 * nll)
            print(f"AIC + BIC score for {kernel_name}: {abic}")
            if abic < best_abic:
                best_abic = abic
                best_kernel = kernel_name
        print(f"best kernel selected by AIC and BIC: {best_kernel}")
        return best_kernel
        
    def evaluate(self, X, return_values_of=['F']):
        '''
        Evaluate mean value, gradient, and Hessian of the objective by normalized X

        Parameters:
            - X ((num_samples, n_var) np.array): the normalized X
            - return_values_of (list): list of return values, choose from 'F', 'dF', 'hF'. Default = ['F']
        Returns:
            - a tuple for each return value array
            - shapes for F, dF, and hF are (num_samples, n_obj), (num_samples, n_obj, n_var), (num_samples, n_obj, n_var, n_var), if num_samples > 1
            - shapes for F, dF, and hF are (n_obj,), (n_obj, n_var), (n_obj, n_var, n_var), if num_samples == 1
        '''
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        F, Std, dF, hF = [], [], [], []
        for _, gp in enumerate(self.gps):
            
            if 'F' in return_values_of:
                K = gp.kernel_(X, gp.X_train_) # K_*^T : shape (N_test, N_train)
                y_mean = K.dot(gp.alpha_) # K_*^T \cdot \alpha : shape (N_test,)
            else:
                y_mean = None

            if 'std' in return_values_of:
                _, y_std = gp.predict(X, return_std=True)
            else:
                y_std = None

            if 'dF' in return_values_of or 'hF' in return_values_of:
                ell = np.exp(gp.kernel_.theta[1:-1]) # $l$, length_scale for each variance : shape (n_var,), 
                sf2 = np.exp(gp.kernel_.theta[0]) # $\sigma_f^2$, length_scale for constant kernel : shape (1,)
                d = np.expand_dims(cdist(X / ell, gp.X_train_ / ell), 2) # $d$: shape (N_test, N_train, 1)
                X_, X_train_ = np.expand_dims(X, 1), np.expand_dims(gp.X_train_, 0) # (N_test, 1, n_var), (1, N_train, n_var)
                dd_N = X_ - X_train_ # numerator : shape (N_test, N_train, n_var)
                dd_D = d * ell ** 2 # denominator : shape (N_test, N_train, n_var)
                dd = safe_divide(dd_N, dd_D) # dd: shape (N_test, N_train, n_var)

                if 'dF' in return_values_of:
                    if   self.gp_kernel[_] == 'Matern12':
                        dK = -sf2 * np.exp(-d) * dd
                    elif self.gp_kernel[_] == 'Matern32':
                        dK = -3 * sf2 * np.exp(-np.sqrt(3) * d) * d * dd
                    elif self.gp_kernel[_] == 'Matern52':
                        dK = -5. / 3 * sf2 * np.exp(-np.sqrt(5) * d) * (1 + np.sqrt(5) * d) * d * dd
                    elif self.gp_kernel[_] == 'RBF':
                        dK = -sf2 * np.exp(-0.5 * d ** 2) * d * dd # dK : shape (N_test, N_train, n_var)
                    dK_T = dK.transpose(0, 2, 1) # dK: shape (N_test, N_train, n_var), dK_T: shape (N_test, n_var, N_train)
                    dy_mean = dK_T @ gp.alpha_ # gp.alpha_: shape (N_train,)
                else:
                    dy_mean = None
                
                if 'hF' in return_values_of:
                    d = np.expand_dims(d, 3) # d: shape (N, N_train, 1, 1)
                    dd = np.expand_dims(dd, 2) # dd: shape (N, N_train, 1, n_var)
                    hd_N = d * np.expand_dims(np.eye(len(ell)), (0, 1)) - np.expand_dims(X_ - X_train_, 3) * dd # numerator
                    hd_D = d ** 2 * np.expand_dims(ell ** 2, (0, 1, 3)) # denominator
                    hd = safe_divide(hd_N, hd_D) # hd: shape (N, N_train, n_var, n_var)
                    if   self.gp_kernel[_] == 'Matern12':
                        hK = -sf2 * np.exp(-d) * (hd - dd ** 2)
                    elif self.gp_kernel[_] == 'Matern32':
                        hK = -3 * sf2 * np.exp(-np.sqrt(3) * d) * (d * hd + (1 - np.sqrt(3) * d) * dd ** 2)
                    elif self.gp_kernel[_] == 'Matern52':
                        hK = -5. / 3 * sf2 * np.exp(-np.sqrt(5) * d) * (-5 * d ** 2 * dd ** 2 + (1 + np.sqrt(5) * d) * (dd ** 2 + d * hd))
                    elif self.gp_kernel[_] == 'RBF': # RBF
                        hK = -sf2 * np.exp(-0.5 * d ** 2) * ((1 - d ** 2) * dd ** 2 + d * hd)
                    hK_T = hK.transpose(0, 2, 3, 1) # hK: shape (N, N_train, n_var, n_var), hK_T: shape (N, n_var, n_var, N_train)
                    hy_mean = hK_T @ gp.alpha_ # hy_mean: shape (N, n_var, n_var)
                else:
                    hy_mean = None
            else:
                dy_mean = None
                hy_mean = None

            F.append(y_mean)
            Std.append(y_std)
            dF.append(dy_mean)
            hF.append(hy_mean)
            
        F   = np.column_stack(F)   if 'F'   in return_values_of else None
        Std = np.column_stack(Std) if 'std' in return_values_of else None
        dF  = np.stack(dF, axis=1) if 'dF'  in return_values_of else None
        hF  = np.stack(hF, axis=1) if 'hF'  in return_values_of else None
        F   = F[0]   if X.shape[0] == 1 and F   is not None else F
        Std = Std[0] if X.shape[0] == 1 and Std is not None else Std
        dF  = dF[0]  if X.shape[0] == 1 and dF  is not None else dF 
        hF  = hF[0]  if X.shape[0] == 1 and hF  is not None else hF
        out = {'F': F, 'std': Std, 'dF': dF, 'hF': hF}
        
        if len(return_values_of) == 1:
            return out[return_values_of[0]]
        else:
            return tuple([out[return_value] for return_value in return_values_of])
            
    def print_summary(self):
        '''
        Prints a summary of the parameters and variables contained in GPs
        '''
        print(f'bounds for scale kernel: {self.scale_kernel_bounds}')
        print(f'bounds for main kernel:  {self.main_kernel_bounds}')
        print(f'bounds for bias kernel:  {self.bias_kernel_bounds}')
        for ig in range(self.real_problem.n_obj):
            print(f'parameters for gp {str(ig)}:')
            print(f'Scale kernel Constant: {self.gps[ig].kernel_.k1.k1.constant_value}')
            print(f'Main kernel length scale: {self.gps[ig].kernel_.k1.k2.length_scale}')
            print(f'Bias kernel Constant: {self.gps[ig].kernel_.k2.constant_value}')

    def print_metric(self, confidence_level=0.9):
        '''
        Prints a summary of metrics for GPs
        '''
        X_nor, Y_nor = self.normalization.do(X=self.X_store, Y=self.Y_store)
        for i, gp in enumerate(self.gps):
            y_true = Y_nor[:, i]
            y_pred, std = gp.predict(X_nor, return_std=True)
            y_pred = y_pred.reshape(-1)
            std = std.reshape(-1)
            # calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            z = get_z_value(confidence_level)
            lower_bound = y_pred - z * std
            upper_bound = y_pred + z * std
            picp = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
            mpiw = np.mean(upper_bound - lower_bound)
            print(f'obj {i}: r2 {r2:.4f} | mae {mae:.4f} | rmse {rmse:.4f} | picp {picp:.4f} | mpiw {mpiw:.4f}')