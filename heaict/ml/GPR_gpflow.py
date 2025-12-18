from heaict.ml.SM_main import SurrogateModel
from heaict.ml.utility import get_z_value, print_with_timestamp

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import tensorflow_probability as tfp
import tensorflow             as tf
import gpflow


class GPR(SurrogateModel):
    '''
    Gaussian process regression surrogate model implemented based on GPFLOW

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
        - X_store ((num_samples, n_var) np.array): stored X data
        - Y_store ((num_samples, n_obj) np.array): stored Y data
    '''
    def __init__(self, 
                 real_problem, 
                 gp_kernel=0.1,
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
        
    def fit(self, X, Y=None, update=False, epochs=300, lr=0.1, optimizer='scipy'):
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
        print(f'Start optimizing the hyperparameters for surrogate model')
        for gp in self.gps:
            if   optimizer == 'scipy':
                self._gpflow_optimize_scipy(gp, epochs)
            elif optimizer == 'tape':
                self._gpflow_optimize(gp, epochs, lr)
        print_with_timestamp(f'Surrogate models constructed successfully')

        self.X_store = X
        self.Y_store = Y
        self.print_metric()
        
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
        for model in self.gps:
            X_tf = tf.convert_to_tensor(X, dtype=tf.float64)
            F_obj, var = model.predict_f(X_tf)
            F_obj = F_obj if 'F' in return_values_of else None
            std = np.sqrt(var) if 'std' in return_values_of else None
            dF_obj = _compute_gradient(X_tf, model).numpy() if 'dF' in return_values_of else None
            hF_obj = _compute_hessian(X_tf, model).numpy() if 'hF' in return_values_of else None
            F.append(F_obj)
            Std.append(std)
            dF.append(dF_obj)
            hF.append(hF_obj)
            
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
            print(f'Scale kernel Constant: {self.gps[ig].kernel.kernels[0].variance.numpy().item()}')
            print(f'Main kernel length scale: {self.gps[ig].kernel.kernels[0].lengthscales.numpy()}')
            print(f'Bias kernel Constant: {self.gps[ig].kernel.kernels[1].variance.numpy().item()}')

    def print_metric(self, confidence_level=0.9):
        '''
        Prints a summary of metrics for GPs
        '''
        X_nor, Y_nor = self.normalization.do(X=self.X_store, Y=self.Y_store)
        X_nor = tf.convert_to_tensor(X_nor, dtype=tf.float64)
        for i, gp in enumerate(self.gps):
            y_true = Y_nor[:, i]
            y_pred, var = gp.predict_f(X_nor)
            y_pred = y_pred.numpy().reshape(-1)
            var = var.numpy().reshape(-1)
            std = np.sqrt(var)
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
            data = (X_nor, Y_nor[:, i: i + 1])
            kernels = {}

            logistic_scaler = tfp.bijectors.Sigmoid(low=tf.constant(self.scale_kernel_bounds[0], dtype=tf.float64), high=tf.constant(self.scale_kernel_bounds[1], dtype=tf.float64))
            logistic_main = tfp.bijectors.Sigmoid(low=tf.constant(self.main_kernel_bounds[0], dtype=tf.float64), high=tf.constant(self.main_kernel_bounds[1], dtype=tf.float64))
            logistic_bias = tfp.bijectors.Sigmoid(low=tf.constant(self.bias_kernel_bounds[0], dtype=tf.float64), high=tf.constant(self.bias_kernel_bounds[1], dtype=tf.float64))
            
            kernels['RBF'] = gpflow.kernels.RBF() + gpflow.kernels.Constant()
            kernels['RBF'].kernels[0].variance = gpflow.Parameter(1.0, transform=logistic_scaler.copy())
            kernels['RBF'].kernels[0].lengthscales = gpflow.Parameter(np.ones(self.real_problem.n_var), transform=logistic_main.copy())
            kernels['RBF'].kernels[1].variance = gpflow.Parameter(1e-2, transform=logistic_bias.copy())
            kernels['Matern12'] = gpflow.kernels.Matern12() + gpflow.kernels.Constant()
            kernels['Matern12'].kernels[0].variance = gpflow.Parameter(1.0, transform=logistic_scaler.copy())
            kernels['Matern12'].kernels[0].lengthscales = gpflow.Parameter(np.ones(self.real_problem.n_var), transform=logistic_main.copy())
            kernels['Matern12'].kernels[1].variance = gpflow.Parameter(1e-2, transform=logistic_bias.copy())
            kernels['Matern32'] = gpflow.kernels.Matern32() + gpflow.kernels.Constant()
            kernels['Matern32'].kernels[0].variance = gpflow.Parameter(1.0, transform=logistic_scaler.copy())
            kernels['Matern32'].kernels[0].lengthscales = gpflow.Parameter(np.ones(self.real_problem.n_var), transform=logistic_main.copy())
            kernels['Matern32'].kernels[1].variance = gpflow.Parameter(1e-2, transform=logistic_bias.copy())
            kernels['Matern52'] = gpflow.kernels.Matern52() + gpflow.kernels.Constant()
            kernels['Matern52'].kernels[0].variance = gpflow.Parameter(1.0, transform=logistic_scaler.copy())
            kernels['Matern52'].kernels[0].lengthscales = gpflow.Parameter(np.ones(self.real_problem.n_var), transform=logistic_main.copy())
            kernels['Matern52'].kernels[1].variance = gpflow.Parameter(1e-2, transform=logistic_bias.copy())

            if select_kernel:
                print(f'Select kernel for obj {str(i)}')
                selected_kernel = self._select_kernel_by_abic(alpha, data, kernels)
                self.gp_kernel.append(selected_kernel)

            kernel = kernels[self.gp_kernel[i]]
            self.gps.append(gpflow.models.GPR(data=data, kernel=kernel))
            
        print(f"initialize surrogate models with kernel: {' '.join(self.gp_kernel)}")

    def _select_kernel_by_abic(self, alpha, data, kernels):
        best_abic = np.inf
        best_kernel = None
        
        for kernel_name, kernel in kernels.items():
            model = gpflow.models.GPR(data=data, kernel=kernel)
            self._gpflow_optimize(model)
            nll = -model.log_marginal_likelihood().numpy()
            num_params = sum([np.prod(p.shape) for p in model.trainable_parameters])
            n = len(data[0])
            abic = (1 - alpha) * (2 * num_params + 2 * nll) + alpha * (np.log(n) * num_params + 2 * nll)
            print(f"AIC + BIC score for {kernel_name}: {abic}") 
            if abic < best_abic:
                best_abic = abic
                best_kernel = kernel_name
        print(f"best kernel selected by AIC and BIC: {best_kernel}")
        return best_kernel
        
    @staticmethod
    def _gpflow_optimize_scipy(model, maxiter=100):
        @tf.function
        def objective_closure():
            return -model.log_marginal_likelihood()
        
        optimizer = gpflow.optimizers.Scipy()
        opt_logs = optimizer.minimize(objective_closure, model.trainable_variables, options=dict(maxiter=maxiter))
    
    @staticmethod
    def _gpflow_optimize(model, epochs=1000, lr=0.01):
        loss_fn = lambda: -model.log_marginal_likelihood()
        optimizer = tf.optimizers.Adam(learning_rate=lr)
        
        @tf.function
        def optimization_step():
            with tf.GradientTape() as tape:
                loss = loss_fn()
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss
        
        for epoch in range(epochs):
            loss = optimization_step()
            

@tf.function(reduce_retracing=True)
def _compute_gradient(X, model):
    with tf.GradientTape() as tape:
        tape.watch(X)
        mean, _ = model.predict_f(X)
    return tape.gradient(mean, X)

@tf.function(reduce_retracing=True)
def _compute_hessian(X, model):
    num_test_points = X.shape[0]
    num_vars = X.shape[1]
    hessians = []
    for i in range(num_test_points):
        with tf.GradientTape(persistent=True) as hessian_tape:
            hessian_tape.watch(X)
            with tf.GradientTape() as gradient_tape:
                gradient_tape.watch(X)
                mean, _ = model.predict_f(X)
            gradient = gradient_tape.gradient(mean, X)
            var_hessian = []
            for j in range(num_vars):
                row_hessian = hessian_tape.gradient(gradient[:, j:j+1], X)
                var_hessian.append(row_hessian[i])
            var_hessian = tf.stack(var_hessian, axis=0)
        hessians.append(var_hessian)
    hessians = tf.stack(hessians, axis=0)
    return hessians