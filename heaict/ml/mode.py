from heaict.ml.layers import fully_connected_layer, multi_head_attention, multi_head_global_attention, multi_head_attention_linear
from heaict.ml.layers import crystal_graph_convolution, graph_attention_convolution, get_activation

import torch
import torch.nn            as nn
from torch_geometric.utils import to_dense_batch, subgraph
from torch_geometric.nn    import global_mean_pool, global_add_pool


class GNN(nn.Module):
    def __init__(
        self,
        node_feat_dim,
        edge_feat_dim,
        hidd_feat_dim,
        conv_model='normal', # 'distance', 'cluster'
        num_conv_sub=3,
        conv_typ_sub='cgc',
        conv_kwargs_sub={},
        num_conv_ads=3,
        conv_typ_ads='cgc',
        conv_kwargs_ads={},
        self_att=False,
        max_num_nodes_sub=64,
        self_att_sub_kwargs={},
        max_num_nodes_ads=7,
        self_att_ads_kwargs={},
        glob_att=False,
        line_att=False,
        att_kwargs={},
        pool='sum',
        fcl_cl_dim=[],
        fcl_en_dim=[],
        num_adsb=5,
        num_site=3,
        norm='batch',
        acti='silu',
        dropout_rate=0,
        classify=False,
        softmaxC=True
    ):
        '''
        Parameters:
            - node_feat_dim (int)
            - edge_feat_dim (int)
            - hidd_feat_dim (int)
            - conv_model (str): Decide the archetype of the model. Default = 'normal'
              'normal': convolutoin slab graph only.
              'distance': convolutoin substrate graph first, then adsb-site graph to 2th site neighbor with metal feature from substrate graph.
              'distance-sg': convolutoin adsb-site graph to 2th site neighbor and slab graph at the same time.
              'cluster': convolutoin substrate graph first, then adsb-site graph to 1th site neighbor with metal feature from substrate graph.
              'cluster-sg': convolutoin adsb-site graph to 1th site neighbor and slab graph at the same time.
              Need node_cluster and node_distance on attributes of slab graphs.
            - num_conv_sub (int): Convolution layer number of slab graphs. Default = 3
            - conv_typ_sub (str): 'cgc' (crystal_graph_convolution ) or 'gat' (graph_attention_convolution). Default = 'cgc'
            - conv_kwargs_sub (dict): Key, value pairs of convolution layers. Check heaict.gnn.layers to get more help. Default = {}
            - num_conv_ads (int): Convolution layer number of adsb-site graphs. Default = 3
            - conv_typ_ads (int): 'cgc' (crystal_graph_convolution ) or 'gat' (graph_attention_convolution). Default = 'cgc'
            - conv_kwargs_ads (dict): Key, value pairs of convolution layers. Check heaict.gnn.layers to get more help. Default = {}
            - self_att (bool): Using self-attention layers after convolution. Default = False
            - max_num_nodes_sub (int): Maximum number of nodes on slag graph. Set if self_att. Default = 64
            - self_att_sub_kwargs (dict): Key, value pairs of slab self-attention layers. Check heaict.gnn.layers to get more help. Default = {}
            - max_num_nodes_ads (int): Maximum number of nodes on adsb-site graph. Set if self_att. Default = 7
            - self_att_ads_kwargs (dict): Key, value pairs of adsb-site self-attention layers. Check heaict.gnn.layers to get more help. Default = {}
            - glob_att (bool): Add global attention after convolutional layers. Default = False
            - line_att (bool): Add linear attention after pooling layers. Default = False
            - att_kwargs (dict): Key, value pairs of global or linear attention layers. Check heaict.gnn.layers to get more help. Default = {}
            - pool (str): pooling layer type. 'sum' or 'mean'. Default = 'sum'
            - fcl_cl_dim (list): dimentions of FCLs on classify prediction.
            - fcl_en_dim (list): dimentions of FCLs on energy prediction.
            - num_adsb (int): number of adsorbate types. Default = 5
            - num_site (int): number of adsorption site types. Default = 3
            - norm (str): batch normal types. 'batch' or 'layer'. Default = 'batch'
            - acti (str): activation function. Default = 'silu'
            - dropout_rate (flaot): dropout rate. Default = 0.
            - classify (bool): Having classification task or not. Default = False
            - softmaxC (bool): Apply softmax to the classification results. Default = True
        '''
        super().__init__()
        # model args 
        self.model_args = {k: v for k, v in locals().items() if k not in ['self', '__class__', 'conv_kwargs']}
        # Embedded layer ----------------------------------------------------
        self.hidd_feat_dim = hidd_feat_dim
        self.node_embed = fully_connected_layer([node_feat_dim, hidd_feat_dim], acti=None, norm=None)
        self.edge_embed = fully_connected_layer([edge_feat_dim, hidd_feat_dim], acti=None, norm=None)
        # Convolution layer ----------------------------------------------------
        self.conv_model = conv_model
        if   conv_typ_sub == 'cgc':
            self.conv_layers_sub = nn.ModuleList([crystal_graph_convolution(hidd_feat_dim, norm=norm, acti=acti, **conv_kwargs_sub) for _ in range(num_conv_sub)])
        elif conv_typ_sub == 'gat':
            self.conv_layers_sub = nn.ModuleList([graph_attention_convolution(hidd_feat_dim, norm=norm, acti=acti, **conv_kwargs_sub) for _ in range(num_conv_sub)])
            
        if conv_model != "normal":
            if conv_typ_ads == 'cgc':
                self.conv_layers_ads = nn.ModuleList([crystal_graph_convolution(hidd_feat_dim, norm=norm, acti=acti, **conv_kwargs_sub) for _ in range(num_conv_ads)])
            elif conv_typ_ads == 'gat':
                self.conv_layers_ads = nn.ModuleList([graph_attention_convolution(hidd_feat_dim, norm=norm, acti=acti, **conv_kwargs_ads) for _ in range(num_conv_ads)])
        else:
            self.register_parameter('conv_layers_ads', None)

        # multi head ----------------------------------------------
        self.self_att = self_att
        if self.self_att:
            self.max_num_nodes_sub = max_num_nodes_sub
            self.self_att_sub  = multi_head_attention(hidd_feat_dim, hidd_feat_dim, hidd_feat_dim, hidd_feat_dim, norm=norm, acti=acti, **self_att_sub_kwargs)
            if conv_model != "normal":
                self.max_num_nodes_ads = max_num_nodes_ads
                self.self_att_ads  = multi_head_attention(hidd_feat_dim, hidd_feat_dim, hidd_feat_dim, hidd_feat_dim, norm=norm, acti=acti, **self_att_ads_kwargs)
            else:
                self.register_parameter('self_att_ads', None)
        else:
            self.register_parameter('self_att_sub', None)
            self.register_parameter('self_att_ads', None)

        # global att ----------------------------------------------------
        self.glob_att = glob_att
        if self.glob_att:
            self.att_glob_sub = multi_head_global_attention(hidd_feat_dim, **att_kwargs)
            if conv_model != "normal":
                self.att_glob_ads = multi_head_global_attention(hidd_feat_dim, **att_kwargs)
            else:
                self.register_parameter('att_glob_ads', None)
        else:
            self.register_parameter('att_glob_sub', None)
            self.register_parameter('att_glob_ads', None)
            
        # pool ----------------------------------------------------
        if   pool == 'mean':
            self.pool_layer = global_mean_pool
            if 'sg' in self.conv_model:
                self.pool_layer_ads = global_mean_pool
            else:
                self.register_parameter('pool_layer_ads', None)
        elif pool == 'sum':
            self.pool_layer = global_add_pool
            if 'sg' in self.conv_model:
                self.pool_layer_ads = global_add_pool
            else:
                self.register_parameter('pool_layer_ads', None)
                
        # linear att ---------------------------------------------
        gfs = 2 if 'sg' in self.conv_model else 1
        self.line_att = line_att
        if self.line_att:
            self.att_inear = multi_head_attention_linear(hidd_feat_dim * gfs, **att_kwargs)
        else:
            self.register_parameter('att_inear', None)
        
        # predict site and adsb ----------------------------------------------------
        self.classify = classify
        self.softmaxC = softmaxC
        if self.classify:
            self.get_adsb  = fully_connected_layer([hidd_feat_dim * gfs] + fcl_cl_dim + [num_adsb], acti=acti, norm=norm, dropout_rate=dropout_rate, predict_cla=True)
            self.get_site  = fully_connected_layer([hidd_feat_dim * gfs] + fcl_cl_dim + [num_site], acti=acti, norm=norm, dropout_rate=dropout_rate, predict_cla=True)
        else:
            self.register_parameter('get_adsb', None)
            self.register_parameter('get_site', None)
        # predict energy ----------------------------------------------------
        self.get_energy = fully_connected_layer([hidd_feat_dim * gfs] + fcl_en_dim + [1], acti=acti, norm=norm, dropout_rate=dropout_rate, predict_reg=True)
    
    def forward(self, data):
        # get data
        node_attr, edge_index, edge_attr, batch, node_cluster, node_distance = data.x, data.edge_index, data.edge_attr, data.batch, data.node_cluster, data.node_distance
        # embed
        node_attr = self.node_embed(node_attr)
        edge_attr = self.edge_embed(edge_attr)
        
        # Convolution
        if   self.conv_model == 'normal':
            for conv_layer in self.conv_layers_sub:
                node_attr = conv_layer(node_attr, edge_index, edge_attr)
            if self.self_att:
                node_attr, mask = to_dense_batch(node_attr, batch, max_num_nodes=self.max_num_nodes_sub)
                node_attr = self.self_att_sub(node_attr, node_attr, node_attr, mask)
            if self.glob_att:
                node_attr = self.att_glob_sub(node_attr, batch)
                
        elif 'sg' not in self.conv_model:
            # substrate data extract
            mask = (node_cluster == 0) | (node_cluster == 1)
            sub_node = mask.nonzero().squeeze()
            sub_batch = batch[sub_node]
            sub_node_attr = node_attr[sub_node].detach().clone()
            sub_edge_index, sub_edge_attr, sub_edge = subgraph(sub_node, edge_index, edge_attr, relabel_nodes=True, return_edge_mask=True, num_nodes=node_attr.size(0))
            sub_edge_attr = sub_edge_attr.detach()
            sub_edge = sub_edge.nonzero().squeeze()
            # substrate convolution            
            for conv_layer in self.conv_layers_sub:
                sub_node_attr = conv_layer(sub_node_attr, sub_edge_index, sub_edge_attr)
            if self.self_att:
                sub_node_attr, mask = to_dense_batch(sub_node_attr, sub_batch, max_num_nodes=self.max_num_nodes_sub)
                sub_node_attr = self.self_att_sub(sub_node_attr, sub_node_attr, sub_node_attr, mask)
                node_attr[sub_node] = sub_node_attr
            if self.glob_att:
                node_attr = self.att_glob_sub(node_attr, batch)
                
            # adss data extract
            if   self.conv_model == 'cluster':
                mask = (node_cluster == 1) | (node_cluster == 2)
            elif self.conv_model == 'distance':
                mask = (node_distance == 0) | (node_distance == 1) | (node_distance == 2)
            ads_node = mask.nonzero().squeeze()
            ads_batch = batch[ads_node]
            ads_node_attr = node_attr[ads_node].detach().clone()
            ads_edge_index, ads_edge_attr, ads_edge = subgraph(ads_node, edge_index, edge_attr, relabel_nodes=True, return_edge_mask=True, num_nodes=node_attr.size(0))
            ads_edge_attr = ads_edge_attr.detach()
            # adss convolution
            for conv_layer in self.conv_layers_ads:
                ads_node_attr = conv_layer(ads_node_attr, ads_edge_index, ads_edge_attr)
            if self.self_att:
                ads_node_attr, mask = to_dense_batch(ads_node_attr, ads_batch, max_num_nodes=self.max_num_nodes_ads)
                ads_node_attr = self.self_att_ads(ads_node_attr, ads_node_attr, ads_node_attr, mask)
            if self.glob_att:
                ads_node_attr = self.att_glob_sub(ads_node_attr, ads_batch)
            
        else:
            if   self.conv_model == 'cluster-sg':
                mask = (node_cluster == 1) | (node_cluster == 2)
            elif self.conv_model == 'distance-sg':
                mask = (node_distance == 0) | (node_distance == 1) | (node_distance == 2)
            ads_node = mask.nonzero().squeeze()
            ads_batch = batch[ads_node]
            ads_node_attr = node_attr[ads_node].detach().clone()
            ads_edge_index, ads_edge_attr, ads_edge = subgraph(ads_node, edge_index, edge_attr, relabel_nodes=True, return_edge_mask=True, num_nodes=node_attr.size(0))
            ads_edge_attr = ads_edge_attr.detach()

            for conv_layer in self.conv_layers_sub:
                node_attr = conv_layer(node_attr, edge_index, edge_attr)
            if self.self_att:
                node_attr, mask = to_dense_batch(node_attr, batch, max_num_nodes=self.max_num_nodes_sub)
                node_attr = self.self_att_sub(node_attr, node_attr, node_attr, mask)
            if self.glob_att:
                node_attr = self.att_glob_sub(node_attr, batch)
            
            for conv_layer in self.conv_layers_ads:
                ads_node_attr = conv_layer(ads_node_attr, ads_edge_index, ads_edge_attr)
            if self.self_att:
                ads_node_attr, mask = to_dense_batch(ads_node_attr, ads_batch, max_num_nodes=self.max_num_nodes_ads)
                ads_node_attr = self.self_att_ads(ads_node_attr, ads_node_attr, ads_node_attr, mask)
            if self.glob_att:
                ads_node_attr = self.att_glob_sub(ads_node_attr, ads_batch)
                
        # pool
        if self.conv_model == 'normal':
            graph_fea = self.pool_layer(node_attr, batch)
        elif 'sg' in self.conv_model:
            graph_fea_sub = self.pool_layer(node_attr, batch)
            graph_fea_ads = self.pool_layer_ads(ads_node_attr, ads_batch)
            graph_fea = torch.cat([graph_fea_sub, graph_fea_ads], dim=-1)
        else:
            graph_fea = self.pool_layer(ads_node_attr, ads_batch)

        # linear att
        if self.line_att:
            graph_fea = self.att_inear(graph_fea)
            
        # predict site and adsb
        if self.classify:
            out_adsb = self.get_adsb(graph_fea)
            out_site = self.get_site(graph_fea)
        # predict energy
        out_energy = self.get_energy(graph_fea)

        # return
        if self.classify:
            if self.softmaxC:
                return out_energy, get_activation('softmax')(out_adsb), get_activation('softmax')(out_site)
            else:
                return out_energy, out_adsb, out_site
        else:
            return out_energy