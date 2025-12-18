import pandas               as pd
import numpy                as np
import matplotlib.pyplot    as plt
from   copy                 import deepcopy
from   decimal              import getcontext, Decimal
from   sklearn.linear_model import LinearRegression, Ridge
from   scipy.optimize       import minimize
import warnings
import os


class HEI():
    '''
    Basic class to solve site perfer on HEI

    Methods:
        1) Read end member thermal data
        - set_end_members:   Initialize and read free energy data of end members
        - plot_fit_result:   Compare the free energy calculated values and fitted values
        - plot_TG_emember:   Function to plot free energy for a set of end members.
        - predict_G:         Using linear regression model to predict free energy
        - get_fit_coeff:     Get the fitted coeff for free energy.
        2) solve site prefer
        - set_composition:   Set alloy composition.
        - solve_site_prefer: Solve the site preference.
        - plot_site_prefer:  Plot the solved site preference.
        - plot_TG_spref:     Plot the free energy of the alloy at the lowest energy (best site occupation).
        - plot_ce_spref:     Plot the configurational entropy of the alloy at the lowest energy (best site occupation).
        - plot_TG_curocp:    Plot the free energy of the alloy at the current site occupation.
    '''
    def __init__(self, site_proportion=[3, 1], consider_entropy=True):
        '''
        Class to solve site preferance on HEI

        Parameters:
            - site_proportion (list): site proportion for A and B sites. Default [3, 1].
            - consider_entropy (bool): whether to consider entropy during solving site preference. Default False.
        '''
        # turn to decimal
        site_proportion[0] = Decimal(str(site_proportion[0]))
        site_proportion[1] = Decimal(str(site_proportion[1]))
        self.site_proportion = site_proportion

        # site prefer data
        self.sp_occp_site1 = None
        self.sp_occp_site2 = None
        self.sp_G = None
        self.sp_T = None
        self.sp_entropy = None

        # consider configurational entropy
        self.consider_entropy = consider_entropy
        
    def set_composition(self, Ecomposition={}, occupy_site1={}, occupy_site2={}):
        '''
        Set alloy composition. You can pass in two, and then the third will be generated automatically. 
        Or give Ecomposition and randomly generate the other two automatically.

        Parameters:
            - Ecomposition (dict): element composition. Default {}
            - occupy_site1 (dict): element occupy at site 1. Default {}
            - occupy_site2 (dict): element occupy at site 2. Default {}
        '''
        if len(occupy_site1) > 0:
            for (ele, sprop) in occupy_site1.items():
                occupy_site1[ele] = Decimal(str(sprop))
        if len(occupy_site2) > 0:
            for (ele, sprop) in occupy_site2.items():
                occupy_site2[ele] = Decimal(str(sprop))
        if len(Ecomposition) > 0:
            for (ele, sprop) in Ecomposition.items():
                Ecomposition[ele] = Decimal(str(sprop))
                
        self.occupy_site1 = occupy_site1
        self.occupy_site2 = occupy_site2
        self.Ecomposition = Ecomposition
        
        # Complete the remaining information if occupy_site1 is known
        if   len(occupy_site1) != 0:
            if   len(occupy_site2) != 0 and len(Ecomposition) == 0:
                self._complete_Ecomp()
            elif len(occupy_site2) == 0 and len(Ecomposition) != 0:
                self._complete_occp()
        elif len(occupy_site1) == 0 and len(occupy_site2) == 0:
            self._random_occp()
        self._check_composition_error()
                    
    def _complete_Ecomp(self):
        '''
        Calculate the total element composition of the alloy based on the element occupation at two sites
        '''
        for ele in self.occupy_site1.keys():
            self.Ecomposition[ele] = \
            self.occupy_site1[ele] * self.site_proportion[0] / sum(self.site_proportion) + \
            self.occupy_site2[ele] * self.site_proportion[1] / sum(self.site_proportion)
    
    def _complete_occp(self):
        '''
        Calculate the element occupation at site 2 based on element occupation at site 1 and element composition of the alloy
        '''
        for ele in self.Ecomposition.keys():
            self.occupy_site2[ele] = \
            (self.Ecomposition[ele] - self.occupy_site1[ele] * self.site_proportion[0] / sum(self.site_proportion)) \
            / (self.site_proportion[1] / sum(self.site_proportion))
        
    def _check_composition_error(self, tolerance=1e-6):
        '''
        Check if element composition and element occupation at site 1 and 2 are all correct
        '''
        tolerance = Decimal(str(tolerance))
        if len(self.occupy_site1) == 0 or len(self.occupy_site2) == 0 or len(self.Ecomposition) == 0:
            raise ValueError('Element composition or site occupancy number is not defined!')
        if any([occp.is_signed() for occp in self.occupy_site1.values()]):
            raise ValueError('One of element occupation on site 1 is negative!')
        if any([occp.is_signed() for occp in self.occupy_site2.values()]):
            raise ValueError('One of element occupation on site 2 is negative!')
        if abs(sum(self.Ecomposition.values()) - Decimal(1.)) > tolerance:
            raise ValueError('Sum of element occupations of the HEA is %f, but it should be 1!' % float(sum(self.Ecomposition.values())))
        if abs(sum(self.occupy_site1.values()) - Decimal(1.)) > tolerance:
            raise ValueError('Sum of element occupations on site 1 is %f, but it should be 1!' % float(sum(self.occupy_site1.values())))
        if abs(sum(self.occupy_site2.values()) - Decimal(1.)) > tolerance:
            raise ValueError('Sum of element occupations on site 2 is %f, but it should be 1!' % float(sum(self.occupy_site2.values())))
        
    def _random_occp(self, occupy_site1_fix={}, show_allocation=False):
        '''
        Generate a random occupation at site 1 and 2 based on element composition of the alloy

        Parameters:
            - occupy_site1_fix (dict): fix some element`s occupation at site 1
            - show_allocation (bool): show the allocation process of site occupation, including the following information. Default False
        '''
        # occp_maxm_bycomp: Based on the composition and the number of sites, calculate the maximum value that each element occupies at site 1, where the maximum is fully occupied, and the minimum is to ensure the correct composition of the HEA (at this time, the occupancy at site 2 is 0).
        # occp_minm_bycomp：Based on the composition and the number of sites, calculate the minimum value that each element occupies at site 1, where the minimum is 0. Additionally, ensure that when site 2 is fully occupied, there is still a portion available for allocation at site 1.
        # occp_maxm_bysite：To ensure that the total occupancy of sites is 1, the amount of occupancy allocated to each element should not exceed the remaining available occupancy.
        # occp_minm_bysite：To ensure that the total occupancy of sites is 1, after each allocation of occupancy, ensure that the remaining available occupancy does not exceed the maximum occupancy that the remaining elements can have.    
        if show_allocation:
            print(' %3s | %4s | %5s - min(%5s, %5s) | %5s - max(%5s, %5s)' % ('Ele', 'occp', 'maxm', 'comp', 'site', 'minm', 'comp', 'site'))
        
        occupy_site1 = {}
        occupy_site2 = {}
            
        occp_maxm_bycomp, occp_minm_bycomp = self._get_occp_maxmin_bycomp()
        
        occp_maxm_drop = {}
        for ele in self.Ecomposition.keys():
            occp_maxm_bysite = Decimal('1') - sum(occupy_site1.values())
            occp_maxm_drop[ele] = occp_maxm_bycomp[ele]
            occp_minm_bysite = Decimal('1') - sum(occupy_site1.values()) - (sum(occp_maxm_bycomp.values()) - sum(occp_maxm_drop.values()))

            occp_maxm = min(occp_maxm_bycomp[ele], occp_maxm_bysite)
            occp_minm = max(occp_minm_bycomp[ele], occp_minm_bysite)

            occp1 = Decimal(str(np.random.rand())) * (occp_maxm - occp_minm) + occp_minm
            if occupy_site1_fix is not None and ele in occupy_site1_fix.keys(): occp1 = Decimal(str(occupy_site1_fix[ele]))
            occupy_site1[ele] = occp1 if len(occupy_site1) != len(self.Ecomposition) - 1 else Decimal('1') - sum(occupy_site1.values())
            occupy_site2[ele] = \
            (self.Ecomposition[ele] - occupy_site1[ele] * self.site_proportion[0] / sum(self.site_proportion)) \
            / (self.site_proportion[1] / sum(self.site_proportion))

            if show_allocation:
                print('  %2s | %4.2f | %5.2f - min(%5.2f, %5.2f) | %5.2f - max(%5.2f, %5.2f)' % (ele, occp1, \
                occp_maxm, occp_maxm_bycomp[ele], occp_maxm_bysite, occp_minm, occp_minm_bycomp[ele], occp_minm_bysite))
        
        self.occupy_site1 = occupy_site1
        self.occupy_site2 = occupy_site2
        self._check_composition_error()

    def _cal_G_atT(self, occp_site1=np.array([]), T=300):
        '''
        Calculate the free energy of a specific occupation at T

        Parameters:
            - occp_site1 (np.array): occupation of each element at site 1
            - T (float): temperature. Default = 300
        '''
        if   len(occp_site1) > 0:
            for i, ele in enumerate(self.Ecomposition.keys()):
                self.occupy_site1[ele] = Decimal(str(occp_site1[i]))
            self._complete_occp()
            site_proportion, occupy_site1, occupy_site2 = self._get_float()
            occp_site2 = np.array([float(op) for op in occupy_site2.values()])
        elif len(occp_site1) == 0:
            site_proportion, occupy_site1, occupy_site2 = self._get_float()
            occp_site1 = np.array([float(op) for op in occupy_site1.values()])
            occp_site2 = np.array([float(op) for op in occupy_site2.values()])
        
        G_em = np.zeros((len(self.Ecomposition), len(self.Ecomposition)))
        for i, site1 in enumerate([ele + str(int(site_proportion[0])) for ele in self.Ecomposition.keys()]):
            for j, site2 in enumerate([ele + str(int(site_proportion[1])) for ele in self.Ecomposition.keys()]):
                G_em[i][j] = self.predict_G(site1 + site2, [T, T, 1], 'end_member_formation')
                                        
        entropy = self._configurational_entropy() if self.consider_entropy else 0
                        
        return np.sum(np.dot(occp_site1.reshape(-1, 1), occp_site2.reshape(-1, 1).T) * G_em) - entropy * T
        
    def _configurational_entropy(self):
        '''
        Calculate the configurational entropy of the current element occupation

        Return:
            - the configurational entropy at J/(mol·atom)/K
        '''
        site_proportion, occupy_site1, occupy_site2 = self._get_float()
        R = 8.314 # J / (mol * K)
        ce = 0
        for ele in self.Ecomposition.keys():
            ce += site_proportion[0] / sum(site_proportion) * occupy_site1[ele] * np.log(occupy_site1[ele]) + \
            site_proportion[1] / sum(site_proportion) * occupy_site2[ele] * np.log(occupy_site2[ele])
        ce = ce * -1 * R
        return ce
    
    def _get_float(self):
        '''
        Get list of float of site proportion and site occupation for further calculation

        Return:
            - lists of floats for site proportion, occupation at site 1, and occupation at site 2 
        '''
        occupy_site1 = self.occupy_site1.copy()
        occupy_site2 = self.occupy_site2.copy()
        for ele in self.Ecomposition.keys():
            occupy_site1[ele] = float(occupy_site1[ele])
            occupy_site2[ele] = float(occupy_site2[ele])
        site_proportion = [float(sp) for sp in self.site_proportion]
        
        return site_proportion, occupy_site1, occupy_site2
    
    def _get_occp_maxmin_bycomp(self):
        '''
        Calculate the maximum and minimum occupation for each element at site 1 based on composition

        Return:
            - two dicts of maximum and minimum occupation at site 1
        '''
        occp_maxm_bycomp = {}
        occp_minm_bycomp = {}
        total_prop = sum(self.site_proportion)
        for ele in self.Ecomposition.keys():
            occp_maxm_bycomp[ele] = min(Decimal(1.), self.Ecomposition[ele] / (self.site_proportion[0] / total_prop))
            occp_minm_bycomp[ele] = max(Decimal(0.), (self.Ecomposition[ele] - self.site_proportion[1] / total_prop) / (self.site_proportion[0] / total_prop))
        
        return occp_maxm_bycomp, occp_minm_bycomp
    
    def _solve_site_prefer_atT(self, T, repeat, maxtry):
        '''
        Solve site preference at T
        '''
        # ignore warning
        warnings.filterwarnings('ignore',category=RuntimeWarning)
        # x bonds
        occp_maxm_bycomp, occp_minm_bycomp = self._get_occp_maxmin_bycomp()
        bonds = []
        for ele in self.Ecomposition.keys():
            bonds.append(tuple([float(occp_minm_bycomp[ele]) + 1e-12, float(occp_maxm_bycomp[ele]) - 1e-12]))
        bonds = tuple(bonds)
        # x constraint
        def constraint(x):
            return sum(x) - 1 + 1e-3
        cons = ({'type':'eq','fun': constraint})
        # try to solve repeat times
        has_solved = False
        min_fun    = np.inf
        best_res   = None
        solve_time = 0
        for i in range(maxtry):
            self._random_occp()
            x0=np.array([float(occp) for occp in list(self.occupy_site1.values())])
            res=minimize(self._cal_G_atT, x0, args=(T,), constraints=cons, method='SLSQP', bounds=bonds, tol=1e-6) # SLSQP, BFGS
            if res.success:
                has_solved = True
                solve_time += 1
                if res.fun < min_fun:
                    best_res = res.x
                    min_fun = res.fun
            if solve_time == repeat:
                break
        # return x to occupy_site1
        if has_solved:
            for i, ele in enumerate(self.Ecomposition.keys()):
                self.occupy_site1[ele] = Decimal(str(best_res[i]))
            self._complete_occp()
        else: # res.x,res.fun
            raise Warning(f'Minimization failure at {T} K: {res.message}')     
    
    def solve_site_prefer(self, Tmin=300, Tmax=1500, Tpoint=50, repeat=6, maxtry=60):
        '''
        Solve the site preference

        Parameters:
            - Tmin (float): minimum temperature. Default = 300
            - Tmax (float): maximum temperature. Default = 1500
            - Tpoint (int): number of T sampling points. Default =50
            - repeat (int): repeat times to search miniest site prefer
        Results:
            - Results will be sotre at self.sp_xxxx
        '''
        Tsequ = np.linspace(Tmin, Tmax, Tpoint)
        sp_occp_site1 = []
        sp_occp_site2 = []
        sp_G = []
        sp_entropy = []
        for T in Tsequ:
            self._solve_site_prefer_atT(T, repeat, maxtry)
            site_proportion, occupy_site1, occupy_site2 = self._get_float()
            sp_occp_site1.append(list(occupy_site1.values()))
            sp_occp_site2.append(list(occupy_site2.values()))
            sp_G.append(self._cal_G_atT(T=T))
            if self.consider_entropy:
                sp_entropy.append(self._configurational_entropy())
        
        self.sp_occp_site1 = np.array(sp_occp_site1)
        self.sp_occp_site2 = np.array(sp_occp_site2)
        self.sp_G = np.array(sp_G)
        self.sp_T = Tsequ
        self.sp_entropy = sp_entropy
    
    def plot_site_prefer(self):
        '''
        Plot the solved site preference. Using this after run solve_site_prefer.
        '''
        for i, ele in enumerate(list(self.Ecomposition.keys())):
            plt.plot(self.sp_T, self.sp_occp_site1[:, i], label=ele + str(self.site_proportion[0]))
        for i, ele in enumerate(list(self.Ecomposition.keys())):
            plt.plot(self.sp_T, self.sp_occp_site2[:, i], label=ele + str(self.site_proportion[1]))
        plt.xlabel('T (K)')
        plt.ylabel('Site occupying fraction')
        plt.legend()
    
    def plot_TG_spref(self):
        '''
        Plot the free energy of the alloy at the lowest energy (best site occupation). Using this after run solve_site_prefer.
        '''
        plt.plot(self.sp_T, self.sp_G)
        plt.xlabel('T (K)')
        plt.ylabel('G (J/mol·atom)')

    def plot_ce_spref(self):
        '''
        Plot the configurational entropy of the alloy at the lowest energy (best site occupation). Using this after run solve_site_prefer.
        '''
        plt.plot(self.sp_T, self.sp_entropy)
        plt.xlabel('T (K)')
        plt.ylabel('Entropy (J/(mol·atom)/K)')

    def plot_TG_curocp(self, Tr=[300, 1500, 50]):
        '''
        Plot the free energy of the alloy at the current site occupation

        Parameters:
            - Tr (list): T range of Tmin, Tmax and T sample point number. Default [300, 1500, 50]
        '''
        T = np.linspace(*Tr)
        G = []
        for t in T:
            G.append(self._cal_G_atT(T=t))
        plt.plot(T, G)
        plt.xlabel('T (K)')
        plt.ylabel('G (J/mol·atom)')

    def _fit_F(self, thermal_data, Natom=1, file_type='qha_gibbs', fit=True, minK=1):
        '''
        Read Gibbs free energy from phonopy and fit a linear relation between free energy and temperature
    
        Parameters:
            - thermal_data (path): the phonopy output file
            - Natom (int): number of atoms on the cell. Default 1
            - file_type (str): type of phonopy output, 'screen_out', 'thermal_yaml', 'qha_gibbs'. Default 'qha_gibbs'
            - fit (bool): use a LinearRegression model to fit T and free energy F. Default True
            - minK (float): minimum K for data collection or fitting. Default 1
        Return:
            - the well fitted LinearRegression model if fit else T and F
        '''
        # read thermal data file
        with open(thermal_data,'r') as f:
            lines = f.readlines()
            T = np.array([])
            F = np.array([])
            if   file_type == 'screen_out':
                # locate data start and end line
                find_start = False
                for i,l in enumerate(lines):
                    if   l == '#      T [K]      F [kJ/mol]    S [J/K/mol]  C_v [J/K/mol]     E [kJ/mol]\n':
                        datal_start = i + 1
                        find_start = True
                    elif l == '\n' and find_start:
                        datal_end = i
                        break
                # extract thermal data
                for line in lines[datal_start: datal_end]:
                    data_in_line = line.strip().split()
                    if float(data_in_line[0]) > minK:
                        T = np.append(T, float(data_in_line[0])) # K
                        F = np.append(F, float(data_in_line[1]) * 1000 / Natom) # kJ/mol -> J/mol-atom
            elif file_type == 'thermal_yaml':
                # find key word
                for i, line in enumerate(lines):
                    if   'temperature' in line and i > 6:
                        data_in_line = line.strip().split()
                        if float(data_in_line[2]) > minK:
                            T = np.append(T, float(data_in_line[2])) # K
                            data_in_line = lines[i + 1].strip().split()
                            F = np.append(F, float(data_in_line[1]) * 1000 / Natom) # kJ/mol -> J/mol-atom
            elif file_type == 'qha_gibbs':
                for line in lines:
                    data_in_line = line.strip().split()
                    if float(data_in_line[0]) > minK:
                        T = np.append(T, float(data_in_line[0])) # K
                        F = np.append(F, float(data_in_line[1]) * 1.602e-19 * 6.02e23 / Natom) # eV -> J -> -> J/mol-atom
        if fit:
            # fit linear model
            x = self._general_x_for_F_fit(T)
            y = F[:,np.newaxis]
            # lr = LinearRegression()
            lr = Ridge(alpha=1.0, solver='svd')
            lr.fit(x, y)
            return lr
        else:
            return T, F

    @staticmethod
    def _general_x_for_F_fit(T):
        '''
        General x for fitting
    
        Parameter:
            - T (array): array of temperature sample points. It is recommended to use np.linespace to general T
        Return:
            - array([Tlog(T), T^2, T^3, T^-1, T])
        '''
        TlnT = T * np.log(T)
        T2 = T ** 2
        T3 = T ** 3
        T_1 = 1/T
        x = np.concatenate((TlnT[:, np.newaxis], T2[:, np.newaxis], T3[:, np.newaxis], T_1[:, np.newaxis], T[:, np.newaxis]), axis=1)
        
        return x

    def predict_G(self, sys, Tr=[300, 1500, 50], typ='metal'):
        '''
        Using linear regression model to predict free energy

        Parameters:
            - sys (str): systems to predict
            - Tr (list): a list of Tmin, Tmax and T point number for plot. Set to T, T ,1 if predict free energy at only one T. Default = [300, 1500, 50]
            - typ (str): type of the system in 'metal', 'end_member_gibbs', 'end_member_formation'. Default 'metal'
        Return:
            - a list or float of the predicted free energy
        '''
        T = np.linspace(*Tr)
        if   typ == 'metal':
            df = self.df_ele
        elif typ == 'end_member_gibbs':
            df = self.df_emg
        elif typ == 'end_member_formation':
            df = self.df_emf
        if   typ == 'metal':
            index = sys
            column = 'lrm'
        else:
            index = sys.split(str(self.site_proportion[0]))[0] + str(self.site_proportion[0])
            column = sys.split(str(self.site_proportion[0]))[1].split(str(self.site_proportion[1]))[0] + str(self.site_proportion[1])
        if Tr[-1] == 1:
            return df.at[index, column].predict(self._general_x_for_F_fit(T)).item()
        else:
            return df.at[index, column].predict(self._general_x_for_F_fit(T)).reshape(-1)
        
    def plot_fit_result(self, thermal_data, Natom=1, file_type='qha_gibbs'):
        '''
        Compare the free energy calculated values and fitted values
    
        Parameters:
            - thermal_data (path): the phonopy output file
            - Natom (int): number of atoms on the cell. Default 1
            - file_type (str): type of phonopy output, 'screen_out', 'thermal_yaml', 'qha_gibbs'. Default 'qha_gibbs'
        '''
        T, F = self._fit_F(thermal_data, Natom, file_type, False)
        Ta = self._general_x_for_F_fit(np.linspace(30, max(T) + 10))
        Ff = self._fit_F(thermal_data, Natom, file_type, True).predict(Ta).squeeze()
        plt.scatter(T, F, c='r',zorder=2, s=1, label='sample point')
        plt.plot(np.linspace(30, max(T) + 10), Ff, c='k', zorder=1, lw=4, label='fitted result')
        plt.xlabel('T (K)')
        plt.ylabel('G (J/mol·atom)')
        plt.legend()

    def set_end_members(self, file_path, file_suffix='', file_type='qha_gibbs', eles=[], Natom_metal=[], Natom_member=4, minK=1):
        '''
        Initialize and read free energy data of end members
        
        Parameters:
            - file_path (path): path where stored the phonopy output files
            - file_suffix (str): file suffix of phonopy output files. Default ''
            - file_type (str): type of phonopy output, 'screen_out', 'thermal_yaml', 'qha_gibbs'. Default 'qha_gibbs'
            - eles (list): list of elements include in the end members. Default []
            - Natom_metal (int): number of atoms on the cell of each metal. Default []
            - Natom_member (int): number of atoms on the cell of each end members. Default 4
            - minK (float): minimum K for data collection or fitting. Default 1
        '''
        # read metal thermal data
        dfm = pd.DataFrame(index=eles, columns=['lrm'])
        for i, ele in enumerate(eles):
            dfm.at[ele, 'lrm'] = self._fit_F(os.path.join(file_path, ele + file_suffix), Natom_metal[i], file_type, True, minK)
        # read end members thermal data
        indexes = [ele + str(self.site_proportion[0]) for ele in eles]
        columns = [ele + str(self.site_proportion[1]) for ele in eles]
        dfeg = pd.DataFrame(index=indexes, columns=columns)
        dfef = pd.DataFrame(index=indexes, columns=columns)
        for index in indexes:
            for column in columns:
                lrg = self._fit_F(os.path.join(file_path, index + column + file_suffix), Natom_member, file_type, True, minK)
                dfeg.at[index, column] = lrg
                T, F = self._fit_F(os.path.join(file_path, index + column + file_suffix), Natom_member, file_type, False, minK)
                minuend = float(self.site_proportion[0]) / float(sum(self.site_proportion)) * dfm.at[index[: -1], 'lrm'].predict(self._general_x_for_F_fit(T)) + \
                float(self.site_proportion[1]) / float(sum(self.site_proportion)) * dfm.at[column[: -1], 'lrm'].predict(self._general_x_for_F_fit(T))
                F = (F - minuend.reshape(-1)).reshape(-1, 1)
                # lrf = LinearRegression()
                lrf = Ridge(alpha=1.0, solver='svd')
                lrf.fit(self._general_x_for_F_fit(T), F)
                dfef.at[index, column] = lrf
        # store fit results
        self.df_ele = dfm
        self.df_emg = dfeg
        self.df_emf = dfef

    def plot_TG_emember(self, sys, Tr=[300, 1600, 100], typ='metal'):
        '''
        Function to plot free energy for a set of end members. Using after set_end_members.

        Parameters:
            - sys (list): list of systems to plot
            - Tr (list): a list of Tmin, Tmax and T point number for plot
            - typ (str): type of the system in 'metal', 'end_member_gibbs', 'end_member_formation'
        '''
        T = np.linspace(*Tr)
        for s in sys:
            plt.plot(T, self.predict_G(s, Tr, typ), label=s)
        plt.xlabel('T (K)')
        plt.ylabel('G (J/mol·atom)')
        plt.legend()

    def get_fit_coeff(self, typ='metal'):
        '''
        Get the fitted coeff for free energy. Using after set_end_members. G(T) = A + BTlnT + CT^2 + DT^3 + ET^-1 + FT

        Parameter:
            - typ (str): type of the system in 'metal', 'end_member_gibbs', 'end_member_formation'
        Return:
            - a DataFrame that stores the coeff information
        '''
        # confirm lr df
        if   typ == 'metal':
            df = self.df_ele
        elif typ == 'end_member_gibbs':
            df = self.df_emg
        elif typ == 'end_member_formation':
            df = self.df_emf
        # get system names
        systems = []
        for index in df.index:
            for column in df.columns:
                if typ == 'metal':
                    systems.append(index)
                else:
                    systems.append(index + column)
        # create empty df
        columns = ['A*1E5', 'B', 'C*1E-3', 'D*1E-7', 'E', 'F']
        dfc = pd.DataFrame(index=systems, columns=columns)
        # add coeff to df
        i = 0
        for index in df.index:
            for column in df.columns:
                dfc.at[systems[i], 'A*1E5'] = df.at[index, column].intercept_.item()
                dfc.loc[systems[i], columns[1:]] = df.at[index, column].coef_
                i += 1
        # number conversion
        dfc['A*1E5'] = dfc['A*1E5'] / 1E5
        dfc['C*1E-3'] = dfc['C*1E-3'] / 1E-3
        dfc['D*1E-7'] = dfc['D*1E-7'] / 1E-7
        dfc['E'] = dfc['E'] / 1E4

        return dfc