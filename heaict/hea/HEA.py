from   pymatgen.core import Element
from   periodictable import elements
from   scipy         import constants
import numpy         as np
import pandas        as pd
import math


df_enthalp = pd.read_csv(f'heaict/hea/Enthalpydata.csv')
kB=constants.physical_constants['Boltzmann constant'][0]
R=constants.physical_constants['molar gas constant'][0]
e=constants.physical_constants['elementary charge'][0]
N_A=constants.Avogadro


class HEA():
    '''
    Class to calculate HEA properties

    Methods:
        - all_property
        - Valence_electron_concentration
        - Atomic_size_difference
        - Electronegativity_difference
        - Mixing_enthalpy
        - Mixing_entropy
        - Correlated_mixing_entropy
    '''
    def __init__(self, hea_com, radius_type='covalent'):
        '''
        Parameter:
            - hea_com (dict): Dictionary of HEA elements and compositions. e.g. {'Ru': 36, 'Fe': 18, ...}
                              Automatically normalizes the composition upon input.
            - radius_type (str): 'covalent' from periodictable or 'metallic' from pymatgen
        '''
        self.hea_com = self._com_normalize(hea_com)
        self.radius_type = radius_type

    def all_property(self, unit='J/mol/K', T=None, radius_type='covalent'):
        '''
        Return all properties

        Parameters:
            - unit (str): Unit for entropy calculation. 'J/mol/K' or 'J/K' or 'eV/atom/K'
            - T (None or float): T for correlated mixing entropy calculation. if None, use average melting point
            
        Return:
            - (n, 3) list with name, value and unit
        '''
        ap = []
        ap.append(['Valence Electron Concentration (VEC)', self.Valence_electron_concentration(), 'e/atom'])
        ap.append(['Atomic size difference (delta)', self.Atomic_size_difference(), '%'])
        ap.append(['Electronegativity difference (chi)', self.Electronegativity_difference(), ''])
        ap.append(['Mixing enthalpy (\Delta H_{mix})', self.Mixing_enthalpy(), 'kJ/mol'])
        ap.append(['Mixing entropy (\Delta S_{mix})', self.Mixing_entropy(unit), unit])
        ap.append(['Correlated_mixing_entropy (S_{coor})', self.Correlated_mixing_entropy(unit, T), unit])
        
        return ap

    def Valence_electron_concentration(self):
        '''
        Valence Electron Concentration (VEC), dimensionless, VEC=\sum_{i=1}^{n}{c_i VEC_i}
        VEC>=8(8.4) -> FCC, VEC<6.87(5.7<=VEC<=7.2) -> BCC
        https://doi.org/10.1016/j.jallcom.2023.170802（https://doi.org/10.1016/j.actamat.2020.03.039）
        '''
        return np.sum([fra * self._get_ele_valence(ele) for ele, fra in self.hea_com.items()])

    def Atomic_size_difference(self):
        '''
        Atomic size difference / mismatch (delta), in %, \delta=100\times\sqrt{\sum_{i=1}^{n}{c_i(1-\frac{r_i}{\overline{r}})^2}}
        delta < 6.6% -> solid solution else amorphous state (https://doi.org/10.1016/j.jallcom.2023.170802)
        '''
        if   self.radius_type == 'metallic':
            ave_rad = np.sum([fra * Element(ele).data['Metallic radius'] for ele, fra in self.hea_com.items()])
            delta   = np.sum([fra * (1 - Element(ele).data['Metallic radius'] / ave_rad)**2 for ele, fra in self.hea_com.items()])
        elif self.radius_type == 'covalent':
            ave_rad = np.sum([fra * getattr(elements, ele).covalent_radius for ele, fra in self.hea_com.items()])
            delta   = np.sum([fra * (1 - getattr(elements, ele).covalent_radius / ave_rad)**2 for ele, fra in self.hea_com.items()])            
        delta   = 100 * np.sqrt(delta)
        return delta

    def Electronegativity_difference(self):
        '''
        Electronegativity difference (chi), dimensionless or e/atom, \Delta\chi=\sqrt{\sum_{i=1}^{n}{c_i(\chi_i-\overline\chi)^2}}
        '''
        ave_eng = np.sum([fra * Element(ele).X for ele, fra in self.hea_com.items()])
        chi     = np.sqrt(np.sum([fra * (Element(ele).X - ave_eng)**2 for ele, fra in self.hea_com.items()]))
        return chi

    def Mixing_enthalpy(self):
        '''
        Mixing enthalpy, kJ/mol, \Delta H_{mix}=4\sum_{i=1,i\neq j}^{n}{c_ic_jH^m_{ij}}
        -11.6 <= H <= 3.2 -> solid solution (https://doi.org/10.1016/j.jallcom.2023.170802)
        '''
        eles = list(self.hea_com.keys())
        fras = list(self.hea_com.values())
        k = 0
        H = 0
        for i in range(len(eles) - 1):
            for j in range(len(eles) - 1 - k):
                H += 4 * self._get_mixing_enthalpy(eles[i], eles[j + 1 + k]) * fras[i] * fras[j + 1 + k]
            k += 1
        return H

    def _Average_enthalpy(self):
        '''
        Average enthalpy, kJ/mol, \overline{H}=\frac{\sum_{i=1,i\neq j}^{n}{c_ic_jH^m_{ij}}}{\sum_{i=1,i\neq j}^{n}{c_ic_j}}
        '''
        eles = list(self.hea_com.keys())
        fras = list(self.hea_com.values())
        k = 0
        H_hat = 0
        coef = 0
        for i in range(len(eles) - 1):
            for j in range(len(eles) - 1 - k):
                H_hat += self._get_mixing_enthalpy(eles[i], eles[j + 1 + k]) * fras[i] * fras[j + 1 + k]
                coef += fras[i] * fras[j + 1 + k]
            k += 1
        return H_hat / coef

    def _Chemical_bond_misfit(self, T):
        '''
        normalized energy fluctuation from chemical bond misfit, dimensionless, 
        $$x_c=2\sqrt{\frac{\sqrt{\sum_i\sum_{j,i\neq j}c_ic_j(H_{ij}-\overline{H})^2}}{k_BT}}$$
        '''
        eles = list(self.hea_com.keys())
        fras = list(self.hea_com.values())
        k = 0
        xc = 0
        H_hat = self._Average_enthalpy()
        for i in range(len(eles) - 1):
            for j in range(len(eles) - 1 - k):
                H_ij = self._get_mixing_enthalpy(eles[i], eles[j + 1 + k])
                xc += fras[i] * fras[j + 1 + k] * (H_ij - H_hat) ** 2
            k += 1
        return 2 * np.sqrt(np.sqrt(xc) * 1e3 / N_A / (kB * T))

    def Mixing_entropy(self, unit='J/mol/K'):
        '''
        Mixing entropy / Configurational entropy, 'J/mol/K' or 'J/K' or 'eV/atom/K', \Delta S_{mix}=-R\sum_{i=1}^{n}{c_i lnc_i}
        S < 1.0 -> low entropy, 1.0 < S < 1.5 -> medium entropy, 1.5 < S -> high entropy, https://doi.org/10.1016/j.heliyon.2024.e26464
        '''
        J2eV = 1 / (N_A * e)
        S = np.sum([fra * np.log(fra) for fra in self.hea_com.values() if fra > 0])
        if   unit == 'J/mol/K':
            S = -1 * R * S
        elif unit == 'J/K':
            S = -1 * kB * S
        elif unit == 'eV/atom/K':
            S = -1 * R * S * J2eV
        return S

    def _Average_Melting_point(self):
        mp_Ru = Element.Ru.data['Melting point']
        if type(mp_Ru) == str:
            has_K = True
        else:
            has_K = False
        if has_K:
            return np.sum([fra * float(Element(ele).data['Melting point'].split(' ')[0]) for ele, fra in self.hea_com.items()])
        else:
            return np.sum([fra * float(Element(ele).data['Melting point']) for ele, fra in self.hea_com.items()])

    def Correlated_mixing_entropy(self, unit='J/mol/K', T=None):
        '''
        Correlated_mixing_entropy, 'J/mol/K' or 'J/K' or 'eV/atom/K', x=x_e+x_c, 
        S_E=k_B\times[1+\frac{x}{2}-ln(x)+ln(1-e^{-x})-\frac{x}{2}\times\frac{1+e^{-x}}{1-e^{-x}}]
        '''
        if T is None:
            T = self._Average_Melting_point()
        S_id = self.Mixing_entropy(unit)
        def get_S_e(x):
            return 1+x/2-np.log(x)+np.log(1-np.exp(-x))-x/2*(1+np.exp(-x))/(1-np.exp(-x))
        S_e  = get_S_e(self._Atom_size_shift(T) + self._Chemical_bond_misfit(T))
        J2eV = 1 / (N_A * e)
        if   unit == 'J/mol/K':
            S_e = R * S_e
        elif unit == 'J/K':
            S_e = kB * S_e
        elif unit == 'eV/atom/K':
            S_e = R * S_e * J2eV
        
        return S_id + S_e

    def _Average_bulk_modulus(self):
        '''
        Average bulk modulus , Pa, \overline{K}=\sum_{i=1}^{n}{c_i K_i}
        '''
        return np.sum([fra * self._get_bulk_module(ele) * 1e9 for ele, fra in self.hea_com.items()])

    def _Average_atomic_volume(self):
        '''
        Average atomic volume, m^3, \overline{V}=\sum_{i=1}^{n}{c_i V_i}
        '''
        def get_ele_volume(ele, radius_type):
            if   radius_type == 'metallic':
                return 4 / 3 * np.pi * Element(ele).data['Metallic radius'] ** 3 * 1e-30
            elif radius_type == 'covalent':
                return 4 / 3 * np.pi * getattr(elements, ele).covalent_radius ** 3 * 1e-30
        return np.sum([fra * get_ele_volume(ele, self.radius_type) for ele, fra in self.hea_com.items()])

    def _Atom_size_shift(self, T):
        '''
        normalized energy fluctuation from atom size shift, dimensionless, 
        x_e=4.12\delta\sqrt{\frac{\overline{K}\overline{V}}{k_BT}}
        '''
        return 4.12 * self.Atomic_size_difference() / 100 * np.sqrt(self._Average_bulk_modulus() * self._Average_atomic_volume() / kB / T)
        
    @staticmethod
    def _com_normalize(hea_com):
        '''
        Automatically normalizes high-entropy alloy composition upon input
        '''
        ary_com      = np.array(list(hea_com.values()))
        ary_com      = np.where(ary_com > 0, ary_com, 1e-10)
        ary_com_nord = ary_com / np.sum(ary_com)
        return {ele: ary_com_nord[_] for _, ele in enumerate(hea_com.keys())}

    @staticmethod
    def _get_ele_valence(ele, valence_oribtals=['s', 'p', 'd']):
        '''
        Count valence number from pymatgen.core.Element.electronic_structure, including orbitals on valence_oribtals
        '''
        VEC = 0
        ele_str = Element(ele).electronic_structure
        for val_orb in ele_str.split('.')[1:]:
            if val_orb[1] in valence_oribtals:
                VEC += int(val_orb[2:])
        return VEC

    @staticmethod
    def _get_bulk_module(ele):
        bm_Ru = Element.Ru.data['Bulk modulus']
        if type(bm_Ru) == str:
            return float(Element(ele).data['Bulk modulus'].split(' ')[0])
        else:
            return float(Element(ele).data['Bulk modulus'])

    @staticmethod
    def _get_mixing_enthalpy(ele1, ele2):
        '''
        Get mixing enthalpy of element 1 and element 2 from data csv
        '''
        mE = df_enthalp[ele1][(df_enthalp['Symbol'] == ele2)].values[0]
        if math.isnan(mE):
            mE = df_enthalp[ele2][(df_enthalp['Symbol'] == ele1)].values[0]
        return float(mE)