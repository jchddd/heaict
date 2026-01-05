from heaict.mobo.utility import find_pareto_front, search_feasible_grid
from pymoo.indicators.hv import HV
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np

def plot_performance_metric(Ys, reference_point=None, color='blue', return_data=False):
    '''
    Plot the performance metric (hypervolume changing with number of evaluations)

    Parameters:
        - Ys (list of np.array): list of Ys or single Y array (target values of sampled points without normalization)
        - reference_point (list of None): reference point for hypervolume calculation. Default = None
        - color (str): Default = blue
        - return_data (bool): return plot data rather than plotting, will return evaluations, hv_mean, and hv_std. Default = False
    '''
    if   not type(Ys) is list:
        Ys = [Ys]
        
    reference_point = np.max(np.vstack(Ys), axis=0)    
    hv_calculator = HV(ref_point=reference_point)
    
    hvs = []
    for Y in Ys:
        hv_list = []
        for i in range(Y.shape[0]):
            hv_list.append(hv_calculator.do(Y[:i + 1,:]))
        hvs.append(hv_list)
    hvs = np.array(hvs)
    
    hv_mean = np.mean(hvs, axis=0)
    if hvs.shape[0] >1: 
        hv_std = np.std(hvs, axis=0)
    else:
        hv_std = 0
    evaluations = np.arange(1, Y.shape[0] + 1)

    if return_data:
        return evaluations, hv_mean, hv_std
    else:
        plt.plot(evaluations, hv_mean, color=color)
        if hvs.shape[0] >1: 
            plt.fill_between(evaluations, hv_mean - hv_std, hv_mean + hv_std, color=color, alpha=0.2)
            
        plt.xlim(0, evaluations[-1])
        plt.title('Hypervolume')
        plt.xlabel('Number of evaluations')
        plt.ylabel('Hypervolume')

def grid_search_objectPS(surrogate_model, interval=0.05, i_obj_x=0, i_obj_y=1, return_pf_grid=False, plot=True):
    '''
    Generate grid points at certain intervals, use the surrogate model to predict the target values of all grid points, 
    and view the distribution of the potential energy surface of the entire target objects.

    Parameters:
        - surrogate_model (class on heaict.mobo.surrogate)
        - interval (float). grid interval on normalized X space. Default = 0.05
        - i_obj_x (int): The index of the objective ouantity located on the X-axis. Default = 0
        - i_obj_y (int): The index of the objective ouantity located on the Y-axis. Default = 1
        - return_pf_grid (bool): Return the grid X at paretor front at transform to real X space. Default = False
        - plot (bool): Show plot or not. Default = True
    '''
    # create grid
    grid = search_feasible_grid(surrogate_model, 0, 1, interval)
    # eval F, find pf
    F_nor = surrogate_model.evaluate(X=grid, return_values_of=['F'])
    F = surrogate_model.normalization.undo(Y=F_nor)
    pf, pfi = find_pareto_front(F)
    # plot
    if plot:
        plt.scatter(F[:, i_obj_x], F[:, i_obj_y], label='Potential energy surface')
        plt.scatter(pf[:, i_obj_x], pf[:, i_obj_y], label='Pareto front')
        plt.legend()
    # return grid
    if return_pf_grid:
        return surrogate_model.normalization.undo(X=grid[pfi])

def plot_cell_sample_2D(b, samples=None):
    '''
    Plot vertices of cell vectors and edges, and project samples into the whole faces of cell vertices

    Parameters:
        - b (heaict.mobo.buffer class)
        - samples (np.array or None): normalized samples
    '''
    fig = plt.figure(figsize=(9, 9))
    
    # plot vertices and edges
    for v in b.cell_vertices:
        plt.scatter(v[0], v[1], c='r')
    plt.scatter(-1, -1, c='r', label='vertices')
    for e in b.edges:
        vs = b.cell_vertices[e[0]]
        ve = b.cell_vertices[e[1]]
        plt.plot([vs[0], ve[0]],[vs[1], ve[1]], c='k--')

    # plot samples
    if samples is not None:
        F = samples - b.origin
        F = F / np.linalg.norm(F, axis=1)[:, np.newaxis]
        sF = np.zeros([0, F.shape[1]])
        for f in F:
            def equations(vars):
                x, y= vars
                eq1 = x / y - f[0] / f[1]
                eq2 = (x-1)**2 + (y-1)**2 - 1
                return [eq1, eq2]
            solution = fsolve(equations, f)
            sF = np.vstack([sF, solution])
        sF = np.where(sF<=1, sF, 1)
        sF = np.where(sF>=0, sF, 0)
        plt.scatter(sF[:, 0], sF[:, 1], color='blue', label='projected samples')

    # set axis property
    plt.axis('equal')
    plt.xlabel('obj 1')
    plt.ylabel('obj 2')
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()

def plot_cell_sample_3D(b, samples=None):
    '''
    Plot vertices of cell vectors and edges, and project samples into the whole faces of cell vertices

    Parameters:
        - b (heaict.mobo.buffer class)
        - samples (np.array or None): normalized samples
    '''
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # plot surface sphere
    if b.cell_shape == 'sphere':
        theta = np.linspace(np.pi, 3 * np.pi / 2, 100)
        phi = np.linspace(np.pi/2, np.pi, 100)
        theta, phi = np.meshgrid(theta, phi)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        ax.plot_surface(x + 1, y + 1, z + 1, color='blue', alpha=0.2, label='1/8sphere')
    elif b.cell_shape == 'triangle':
        x = np.linspace(0, 1, 101)
        y = np.linspace(0, 1, 101)
        x, y = np.meshgrid(x, y)
        z = 1 - x - y
        mask = z >= 0
        x = x[mask]
        y = y[mask]
        z = z[mask]
        ax.plot_trisurf(x, y, z, color='blue', alpha=0.2, label='triangle')
        
    # plot vetices and edges
    ax.scatter(b.cell_vertices[:, 0], b.cell_vertices[:, 1], b.cell_vertices[:, 2], color='red', label='vertices')
    for e in b.edges:
        vs = b.cell_vertices[e[0]]
        ve = b.cell_vertices[e[1]]
        ax.plot([vs[0], ve[0]], [vs[1], ve[1]], [vs[2], ve[2]], 'k--')

    # plot samples
    if samples is not None:
        F = samples - b.origin
        F = F / np.linalg.norm(F, axis=1)[:, np.newaxis]
        sF = np.zeros([0, F.shape[1]])
        if b.cell_shape == 'sphere':
            for f in F:
                def equations(vars):
                    x, y, z = vars
                    eq1 = x / y - f[0] / f[1]
                    eq2 = x / z - f[0] / f[2]
                    eq3 = (x-1)**2 + (y-1)**2 + (z-1)**2 - 1
                    return [eq1, eq2, eq3]
                solution = fsolve(equations, f)
                sF = np.vstack([sF, solution])
        elif b.cell_shape == 'triangle':
            for f in F:
                def equations(vars):
                    x, y, z = vars
                    eq1 = x / y - f[0] / f[1]
                    eq2 = x / z - f[0] / f[2]
                    eq3 = x + y + z - 1
                    return [eq1, eq2, eq3]
                solution = fsolve(equations, f)
                sF = np.vstack([sF, solution])
        sF = np.where(sF<=1, sF, 1)
        sF = np.where(sF>=0, sF, 0)
        ax.scatter(sF[:, 0], sF[:, 1], sF[:, 2], color='blue', label='projected samples')

    # set axis property
    plt.axis('equal')
    plt.xlabel('obj 1')
    plt.ylabel('obj 2')
    ax.set_zlabel('obj 3')
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ax.set_zlim([0, 1])
    ax.view_init(elev=30, azim=45)
    plt.show()
