import torch
import torch.nn            as nn
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn    import global_mean_pool
from hgcode.layers         import crystal_graph_convolution
from hgcode.layers         import graph_attention_convolution
from hgcode.layers         import multi_head_attention
from hgcode.layers         import multi_head_attention_linear
from hgcode.layers         import fully_connected_layer
from hgcode.layers         import cluster_pool
from hgcode.layers         import multi_head_global_attention


class gnn(nn.Module):
    def __init__(
        self,
        node_feat_dim_init,
        edge_feat_dim_init,
        node_feat_dim_hidd,
        edge_feat_dim_hidd,
        max_num_nodes,
        fcl_e_dims=[],
        fcl_c_dims=[],
        num_adsb=5,
        num_site=3,
        num_conv=3,
        typ_conv='gat',
        gatt='self', # 'self', 'glob', 'none'
        pool='cluster', # 'cluster', 'mean'
        fcla=True,
        acti='tanh',
        norm='batch',
        dropout_rate=0,
        classify=True,
        **conv_kwargs
    ):
        '''
        Parameters:
            - node_feat_dim (int): node feature dimension.
            - edge_feat_dim (int): edge feature dimension.
            - hidd_feat_dim (int): hidden feature dimension.
            - max_num_nodes (int): the maximum number of nodes in a single graph.
            - fcl_e_dims (list): dimensional sequence of the hidden layer of FCL for predicting adsorption energy. Default = []
            - fcl_c_dims (list): dimensional sequence of the hidden layer of FCLs used to classify adsorption site and adsorbate. Default = []
            - num_adsb (int): number of classes of adsorbate. Default = 5
            - num_site (int): number of classes of adsorption site. Default = 4
            - num_conv (int): number of convolution layers. Default = 3
            - type_conv (str): convolution layer, 'gat' or 'cgc'. Default = 'gat'
            - gatt (str): type of attention block after convolution layers, 'self' or 'glob'. Default = 'self'
            - pool (str): type of pooling layer, 'cluster' or 'mean'. Default = 'mean'
            - fcla (bool): add attention layer to FCLs. Default = True
            - acti (str): activation function. Default = 'silu'
            - norm (str): normalization function. Default = 'batch'
            - dropout_rate (float): dropout rate for FCL. Default = 0
            - classify (bool): whether to perform classification tasks for adsorption site and adsorbate. Default = True
            - **conv_kwargs: other key parameters for convolution layer.
        '''
        super().__init__()
        # model args
        self.model_args = {k: v for k, v in locals().items() if k not in ['self', '__class__', 'conv_kwargs']}
        self.model_args.update(conv_kwargs)
        # variable
        hidd_feat_dim = node_feat_dim_hidd
        self.hidd_feat_dim = hidd_feat_dim
        self.max_num_nodes = max_num_nodes
        # Embedded layer
        self.node_embed = fully_connected_layer([node_feat_dim_init, node_feat_dim_hidd], acti=None, norm=None)
        self.edge_embed = fully_connected_layer([edge_feat_dim_init, edge_feat_dim_hidd], acti=None, norm=None)
        # Convolution layer
        if   typ_conv.lower() == 'cgc':
            self.conv_layers = nn.ModuleList([crystal_graph_convolution(node_feat_dim_hidd, edge_feat_dim_hidd, norm=norm, acti=acti, **conv_kwargs) for _ in range(num_conv)])
        elif typ_conv.lower() == 'gat':
            self.conv_layers = nn.ModuleList([graph_attention_convolution(node_feat_dim_hidd, edge_feat_dim_hidd, node_feat_dim_hidd, norm=norm, acti=acti, **conv_kwargs) for _ in range(num_conv)])
        # self-attention
        self.gatt = gatt
        if   self.gatt == 'self': 
            self.self_att = multi_head_attention(hidd_feat_dim, hidd_feat_dim, hidd_feat_dim, hidd_feat_dim, 2)
        elif self.gatt == 'glob':
            self.glob_att = multi_head_global_attention(hidd_feat_dim, [hidd_feat_dim])
        # pooling
        self.pool = pool
        if   pool == 'cluster':
            class_feat_dim = hidd_feat_dim * 2
            reger_feat_dim = hidd_feat_dim * 3
            
        elif pool == 'mean':
            class_feat_dim = hidd_feat_dim
            reger_feat_dim = hidd_feat_dim
        # classify
        self.classify = classify
        if classify:
            self.get_adsb   = fully_connected_layer([class_feat_dim] + fcl_c_dims + [num_adsb], acti=acti, norm=norm, dropout_rate=dropout_rate, predict_cla=True)
            self.get_site   = fully_connected_layer([class_feat_dim] + fcl_c_dims + [num_site], acti=acti, norm=norm, dropout_rate=dropout_rate, predict_cla=True)
        else:
            self.register_parameter('get_adsb', None)
            self.register_parameter('get_site', None)
        # regre
        self.fcla = fcla
        if   pool == 'cluster':
            self.to_one_hidden = fully_connected_layer([reger_feat_dim, hidd_feat_dim], acti=acti, norm=norm)
        if   fcla:
            self.multi_att = multi_head_attention_linear(hidd_feat_dim, 3, [hidd_feat_dim], acti=acti, norm=norm)
        self.get_energy = fully_connected_layer([hidd_feat_dim] + fcl_e_dims + [1], acti=acti, norm=norm, dropout_rate=dropout_rate, predict_reg=True)
    
    def forward(self, data):
        # get data
        node_attr, edge_index, edge_attr, batch, node_cluster = data.x, data.edge_index, data.edge_attr, data.batch, data.node_cluster
        # embed
        node_attr = self.node_embed(node_attr)
        edge_attr = self.edge_embed(edge_attr)
        # conv
        for conv_layer in self.conv_layers:
            node_attr = conv_layer(node_attr, edge_index, edge_attr)
        # att
        if   self.gatt == 'self':
            node_attr, mask = to_dense_batch(node_attr, batch, max_num_nodes=self.max_num_nodes)
            node_attr = self.self_att(node_attr, node_attr, node_attr, mask)
            node_attr = node_attr.reshape(-1, self.hidd_feat_dim)[mask.reshape(-1)]
        elif self.gatt == 'glob':
            node_attr = self.glob_att(node_attr, batch)
        elif self.gatt == 'none':
            pass
        # pooling
        if   self.pool == 'cluster':
            graph_fea = cluster_pool(node_attr, node_cluster, batch)
        elif self.pool == 'mean':
            graph_fea = global_mean_pool(node_attr, batch)
        # classify
        if   self.pool == 'cluster' and self.classify:
            out_adsb = self.get_adsb(graph_fea[:, self.hidd_feat_dim:])
            out_site = self.get_site(graph_fea[:, self.hidd_feat_dim:])
        elif self.pool == 'mean' and self.classify:
            out_adsb = self.get_adsb(graph_fea)
            out_site = self.get_site(graph_fea)
        # regre
        if self.pool == 'cluster':
            graph_fea = self.to_one_hidden(graph_fea)
        if self.fcla:
            graph_fea = self.multi_att(graph_fea)
        out_energy = self.get_energy(graph_fea)
        # return
        if self.classify:
            return out_energy, out_adsb, out_site
        else:
            return out_energy
    
    def count_parameters(self, print_info=False):
        if print_info:
            for name, module in self.named_children():
                params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                print(f"{name}: Number of trainable parameters: {params}")
                trainable_variables = [param for param in module.parameters() if param.requires_grad]
                for var_name, var_param in module.named_parameters():
                    if var_param.requires_grad:
                        print(f"  - {var_name} (size: {var_param.size()})")
        return sum(p.numel() for p in self.parameters())