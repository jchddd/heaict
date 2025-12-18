from heaict.cats.surface import brute_force_surface, get_activity_selectivity
from heaict.cats.utility import round_preserve_sum
from heaict.mobo.utility import print_with_timestamp
from heaict.mobo.problem import Real_Problem
from heaict.ml.utility import fix_seed
from heaict.hea.HEA import HEA
from heaict.hea.HEI import HEI
from heaict.ml.mode import GNN
from heaict.para import *

from copy import deepcopy
import numpy as np
import concurrent
import pickle
import torch
import os


class HEA_problem(Real_Problem):
    def __init__(self, size, archetype, model, eles=[], batch_size=5000, seed=666, max_workers=6, n_var=4, n_obj=2, save_result=False, save_path='./'):
        '''
        Problem to predict activity and FE of HEAs

        Parameters:
            - size (int): surface size
            - archetype (path): path to archetype files
            - model (path): path to state dict of GNN
            - eles (list): list of elements. Default = []
            - batch_size (int). batch size when perform energy prediction. Default = 5000
            - seed (None or int): give a number to fix random seed. Default = None
            - max_workers (int): number of max workers when parallel. Default = 6
            - n_var (int): number of variables. Default = 4
            - n_obj (int): number of objectives. Default = 2. Choose from 2 (activity and selectivity) or 3 (add stability)
            - save_result (bool): save results from net energies. Default = False
            - save_path (path). Default = './'
        '''
        super().__init__(n_var, n_obj, 4, 0.05, 0.35)
        self.size = size
        self.archetype = archetype
        self.model = model
        self.eles = eles
        self.batch_size = batch_size
        self.seed = seed
        self.max_workers = max_workers

        self.save_result = save_result
        self.save_path = save_path

        
    def evaluate_objective_batch(self, X):
        # load model
        model = GNN(**para_model)
        model.load_state_dict(torch.load(self.model, map_location=torch.device('cpu')))
        # create bfs
        bfs = brute_force_surface(surface_size=(self.size, self.size), n_neighbors=3, para_grab_feature=para_grab_feature)
        bfs.read_archetype(self.archetype, zero_frac, grid_basis_in_frac, para_constr_graph={}, para_infer_site={}, para_add_feat=para_add_feat)
        # initial variable
        F = []
        hea_names = []
        surfaces = []
        Gms = []
        # create bfs 
        for i, x in enumerate(X):
            if self.seed is not None:
                fix_seed(self.seed)
            # confirm element fraction
            elef = [x[0], x[1], x[2], x[3], 1-sum(x)]
            elef = round_preserve_sum(elef)
            ele_fra = {k: v for k, v in zip(self.eles, elef)}
            # print hea composition
            hea_name = "-".join([key+str(round(value, 2)) for key, value in ele_fra.items()])
            print_with_timestamp(f'hea {hea_name}')
            hea_names.append(hea_name)
            hea = HEA(ele_fra)
            Gms.append(hea.Mixing_enthalpy() - 1200 * (hea.Correlated_mixing_entropy() * 1e-3))
            # create hea surfaces
            hea_surface=deepcopy(bfs)
            hea_surface.index = i
            hea_surface.create_slab_HEA(ele_fra)
            hea_surface.eval_gross_energies(model, batch_size=self.batch_size)
            surfaces.append(hea_surface)
        # get net energy
        print_with_timestamp(f'get net energy')
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(slab.get_net_energies, reaction_steps) for slab in surfaces]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        # get activity and selectivity
        for index, result in results:
            if self.save_result:
                with open(os.path.join(self.save_path, f'{hea_names[index]}.pkl'), 'wb') as f:
                    pickle.dump(result, f)
            kNRR, kHER, FE, U = get_activity_selectivity(result)
            if   self.n_obj == 2:
                F.append([index, -1*np.log(kNRR), -FE])
                print_with_timestamp(f'index {index} kNRR {np.log(kNRR)} FE {FE} U {U}')
            elif self.n_obj == 3:
                Gm = Gms[index]
                F.append([index, -1*np.log(kNRR), -FE, Gm])
                print_with_timestamp(f'index {index} kNRR {np.log(kNRR)} FE {FE} Gm {Gm} U {U}')
        del surfaces
        # sort F
        F = np.array(F)
        sort_idx = np.argsort(F[:,0])
        F = F[sort_idx][:, 1:]
        return F
        
    def evaluate_constraint_batch(self, X):
        G = []
        for x in X:
            # constrain for the final ele
            xi = 1 - sum(x)
            g1 = -xi + 0.05
            g2 = xi - 0.35
            # constrain for FCC structure
            elef = [x[0], x[1], x[2], x[3], 1-sum(x)]
            ele_fra = {k: v for k, v in zip(self.eles, elef)}
            hea = HEA(ele_fra)
            g3 = hea.Atomic_size_difference() - 6.6
            # FCC fration > 0.5
            g4 = -np.sum(x[:3]) + 0.5
            G.append([g1, g2, g3, g4])
        return np.array(G)

    def add_Gm(self, X, Y):
        '''
        Add Gibbs free energy (stability) to a Bi-objective results 

        Parameters:
            - X ((n_sample, n_var) array): Alloy compositoins.
            - Y ((n_sample, 2) array): Bi-objective results (activity and selectivity).
        '''
        Gms = []
        for x in X:
            elef = [x[0], x[1], x[2], x[3], 1-sum(x)]
            elef = round_preserve_sum(elef)
            ele_fra = {k: v for k, v in zip(self.eles, elef)}
            hea = HEA(ele_fra)
            Gms.append(hea.Mixing_enthalpy() - 1200 * (hea.Correlated_mixing_entropy() * 1e-3))
        Y = np.column_stack([Y, np.array(Gms).reshape(-1, 1)])
        return Y


class HEI_problem(Real_Problem):
    def __init__(self, size, archetype, model, thermal, thermal_para=(), eles=[], batch_size=5000, seed=None, max_workers=6, n_var=4, n_obj=2, save_result=False, save_path='./'):
        '''
        Problem to predict activity and FE of HEAs

        Parameters:
            - size (int): surface size
            - archetype (path): path to archetype files
            - model (path): path to state dict of GNN
            - thermal (path): path to thermal data
            - thermal_para (tuple of list): eles and Natom_metal for thermal data reading
            - eles (list): list of elements. Default = []
            - batch_size (int). batch size when perform energy prediction. Default = 5000
            - seed (None or int): give a number to fix random seed. Default = None
            - max_workers (int): number of max workers when parallel. Default = 6
            - n_var (int): number of variables. Default = 4
            - n_obj (int): number of objectives. Default = 2. Choose from 2 (activity and selectivity) or 3 (add stability)
            - save_result (bool): save results from net energies. Default = False
            - save_path (path). Default = './'
        '''
        super().__init__(n_var, n_obj, 3, 0.05, 0.35)
        self.size = size
        self.archetype = archetype
        self.model = model
        self.eles = eles
        self.batch_size = batch_size
        self.seed = seed
        self.max_workers = max_workers

        self.save_result = save_result
        self.save_path = save_path

        self.site_pre = HEI([3, 1])
        self.site_pre.set_end_members(thermal, '.dat', eles=thermal_para[0], Natom_metal=thermal_para[1])
        
    def evaluate_objective_batch(self, X):
        # load model
        model = GNN(**para_model)
        model.load_state_dict(torch.load(self.model, map_location=torch.device('cpu')))
        # create bfs
        bfs = brute_force_surface(surface_size=(self.size, self.size), n_neighbors=3, para_grab_feature=para_grab_feature)
        bfs.read_archetype(self.archetype, zero_frac, grid_basis_in_frac, para_constr_graph={}, para_infer_site={}, para_add_feat=para_add_feat)
        # initial variable
        F = []
        hei_names = []
        surfaces = []
        Gms = []
        # create bfs 
        for i, x in enumerate(X):
            if self.seed is not None:
                fix_seed(self.seed)
            # confirm element fraction
            elef = [x[0], x[1], x[2], x[3], 1-sum(x)]
            elef = round_preserve_sum(elef)
            ele_fra = {k: v for k, v in zip(self.eles, elef)}
            # print hei composition
            self.site_pre.set_composition(ele_fra)
            self.site_pre.solve_site_prefer(1200, 1200, 1, repeat=10)
            Gms.append(self.site_pre.sp_G[0])
            spA = {k: float(v) for k, v in self.site_pre.occupy_site1.items()}
            spB = {k: float(v) for k, v in self.site_pre.occupy_site2.items()}
            hei_name = "-".join([key+str(round(value, 2)) for key, value in ele_fra.items()]) + '_A_' + "-".join([key+str(round(value, 2)) for key, value in spA.items()]) + '_B_' + "-".join([key+str(round(value, 2)) for key, value in spB.items()])
            print_with_timestamp(f'{hei_name}')
            hei_names.append(hei_name)
            # create hea surfaces
            hei_surface=deepcopy(bfs)
            hei_surface.index = i
            hei_surface.create_slab_HEI({'A': spA, 'B': spB})
            hei_surface.eval_gross_energies(model, batch_size=self.batch_size)
            surfaces.append(hei_surface)
        # get net energy
        print_with_timestamp(f'get net energy')
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(slab.get_net_energies, reaction_steps) for slab in surfaces]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        # get activity and selectivity
        for index, result in results:
            if self.save_result:
                with open(os.path.join(self.save_path, f'{hei_names[index]}.pkl'), 'wb') as f:
                    pickle.dump(result, f)
            kNRR, kHER, FE, U = get_activity_selectivity(result)
            if   self.n_obj == 2:
                F.append([index, -1*np.log(kNRR), -FE])
                print_with_timestamp(f'index {index} kNRR {np.log(kNRR)} FE {FE} U {U}')
            elif self.n_obj == 3:
                Gm = Gms[index]
                F.append([index, -1*np.log(kNRR), -FE, Gm])
                print_with_timestamp(f'index {index} kNRR {np.log(kNRR)} FE {FE} Gm {Gm} U {U}')
        del surfaces
        # sort F
        F = np.array(F)
        sort_idx = np.argsort(F[:,0])
        F = F[sort_idx][:, 1:]
        return F
        
    def evaluate_constraint_batch(self, X):
        G = []
        for x in X:
            # constrain for the final ele
            xi = 1 - sum(x)
            g1 = -xi + 0.05
            g2 = xi - 0.35
            # constrain for FCC structure
            elef = [x[0], x[1], x[2], x[3], 1-sum(x)]
            ele_fra = {k: v for k, v in zip(self.eles, elef)}
            hea = HEA(ele_fra)
            # FCC fration > 0.5
            g3 = -np.sum(x[:3]) + 0.5
            G.append([g1, g2, g3])
        return np.array(G)

    def add_Gm(self, X, Y):
        '''
        Add Gibbs free energy (stability) to a Bi-objective results 

        Parameters:
            - X ((n_sample, n_var) array): Alloy compositoins.
            - Y ((n_sample, 2) array): Bi-objective results (activity and selectivity).
        '''
        Gms = []
        for x in X:
            elef = [x[0], x[1], x[2], x[3], 1-sum(x)]
            elef = round_preserve_sum(elef)
            ele_fra = {k: v for k, v in zip(self.eles, elef)}
            self.site_pre.set_composition(ele_fra)
            self.site_pre.solve_site_prefer(1200, 1200, 1, repeat=10)
            Gms.append(self.site_pre.sp_G[0])
        Y = np.column_stack([Y, np.array(Gms).reshape(-1, 1)])
        return Y