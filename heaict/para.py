import numpy as np

para_grab_feature = {
    'features': ['number', 'row', 'group', 'atomic_radius', 'atomic_mass', 'X', 'electron_affinity', 'ionization_energy', 'valence', 'Atomic orbitals'], 
    'use_onehot':True, 
    'onehot_dim':10, 
    'compress_onehot':True
}

para_add_feat = {
    'distance_classes': 4, 
    'distance_feature':True, 
    'use_NRRBOP':True, 
    'coordinate':[9, 12], 
    'bond_feature': False, 
    'add_node_distance': True
}

thermal_para = (['Mo', 'Mn', 'Fe', 'Ru', 'Co', 'Pd', 'Ni', 'Cu'], [2, 2, 2, 2, 2, 4, 4, 4])

zero_frac = (0.4444, 0.4444, 0.3)

grid_basis_in_frac = np.array([(0.1111, 0, 0), (0, 0.1111, 0), (-0.1111/3, -0.2222/3, -0.0992)])
slab_lattice = np.array([(23.173, 0, 0), (-11.586, 20.068, 0), (0, 0, 21.354)])

para_model = {
    'node_feat_dim': 68, 'edge_feat_dim': 18,
    'hidd_feat_dim': 18, 'conv_model': 'cluster-sg',
    'num_conv_sub': 3, 'conv_typ_sub': 'cgc',
    'conv_kwargs_sub': {'cat_edge': True, 'residual': True},
    'num_conv_ads': 3, 'conv_typ_ads': 'cgc',
    'conv_kwargs_ads': {'cat_edge': True, 'residual': True},
    'self_att': False, 'line_att': False, 'glob_att': True, 'att_kwargs': {}, 
    'pool': 'sum','fcl_cl_dim': [60, 30], 'fcl_en_dim': [60, 30, 15], 'acti': 'gelu','classify': True, 'num_adsb': 7
}

para_trainer = { # 683
    'epoch': 200,
    'batch': 256,
    'metric': ['mae', 0.99, 7, 3, 0.06],
    'optimizer': ['AdamW', 0.001, 1e-5],
    'scheduler': ['ConstantLR'],
    'max_norm': 10.0
}

reaction_steps=[
    ['N2', 'NNH', 'NH', 'NH3', 'H'],
    ['NNH', 'N2', 'NH', 'NH3', 'H'],
    ['H', 'N2', 'NNH', 'NH', 'NH3']
]