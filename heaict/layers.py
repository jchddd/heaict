import math
import torch
import torch.nn               as nn
import torch.nn.functional    as F
from torch                    import Tensor
from torch_geometric.utils    import softmax as graph_softmax
from torch_geometric.utils    import scatter
from torch_geometric.nn.conv  import MessagePassing
from torch_geometric.nn.inits import glorot, zeros


class crystal_graph_convolution(MessagePassing):
    '''
    Crystal graph convolution layer which mainly use a Gate MLP (sigmoid(x_i|x_j|ea) * acti_core(x_i|x_j|ea)) to aggregated neighbor information
    '''
    def __init__(self, 
                 node_feat_dim,
                 edge_feat_dim,
                 hidd_mlp_dim=[],
                 acti_core='softplus',
                 acti='tanh',
                 norm='batch', 
                 bias=True, 
                 weight_by_edge=False,
                 output_mlp=False,
                 residual=True,
                 **kwargs):
        '''
        Parameters:
            - node_feat_dim (int): node(atom) feature dimension.
            - edge_feat_dim (int): edge(bond) feature dimension.
            - hidd_mlp_dim (array-like): hidden layers when transform x_i|x_j|ea to shape of x_i. Default []
            - acti_core (str): activation function for Gate MLP. Default 'softplus'
            - acti (str): activation function of all linear layers. Default 'silu'
            - norm (str or None): normalization layer for final aggregated features x_aggr. Default 'batch'
            - bias (bool): whether adds bias to all linear layers. Default True
            - weight_by_edge (bool): weight the message feature by element-wise multiplication with weights obtained by linear transfrom of edge_attr. Default False
            - output_mlp (bool): add a linear layer for aggregated features x with same demension of x_aggr. Default False
            - residual (bool): add the unaggregated feature x to the aggregated feature x_aggr as the output. Default True
        '''
        super().__init__(aggr='add', flow='target_to_source', **kwargs)
        self.model_args = {k: v for k, v in locals().items() if k not in ['self', '__class__', 'kwargs']}
        self.model_args.update(kwargs)
        # param
        self.norm                   = norm
        self.weight_by_edge         = weight_by_edge
        self.output_mlp             = output_mlp
        self.residual               = residual
        # dimension and activational function for fcls
        dim_sequence                = [node_feat_dim * 2 + edge_feat_dim] + hidd_mlp_dim + [node_feat_dim]
        self.sqeuence_core          = fully_connected_layer(dim_sequence, acti_core, norm, bias=bias)
        self.sqeuence_gate          = fully_connected_layer(dim_sequence, 'sigmoid', norm, bias=bias)
        # whether add weight from edge
        if self.weight_by_edge:
            self.weight_edge        = fully_connected_layer([edge_feat_dim, node_feat_dim], acti, 'layer', bias=False)
        else:
            self.register_parameter('weight_edge', None)
        # whether additional transform output matrix
        if self.output_mlp:
            self.linear_output      = fully_connected_layer([node_feat_dim, node_feat_dim], acti, norm, bias=bias)
        else:
            self.register_parameter('linear_output', None)
        # normalization layer
        self.layer_norm             = get_normalization(norm, node_feat_dim)
        
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        # cat x_i, x_j, and edge_attr to z
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        # z go through core and gate fcls
        x_message = self.sqeuence_core(z) * self.sqeuence_gate(z)
        # weight by edge
        if self.weight_by_edge:
            weight = self.weight_edge(edge_attr)
            x_message = x_message * weight
            
        return x_message
    
    def update(self, aggr_out, x):
        x_update = aggr_out
        # additional fcl 
        if self.output_mlp:
            x_update = self.linear_output(x_update)
        # residual 
        if self.residual:
            x_update = x_update + x
        # normalization
        x_update = x_update if self.norm is None else self.layer_norm(x_update)
        
        return x_update
    
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
        
    def set_requires_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = requires_grad

            
class graph_attention_convolution(MessagePassing):
    '''
    Graph attention convolution layer which calculate attentions for neighbors of each node by a weight matrix W and a weight vector a
    '''
    def __init__(self, 
                 node_feat_dim,
                 edge_feat_dim,
                 hidd_feat_dim,                 
                 num_heads=3, 
                 acti='tanh', 
                 norm='batch', 
                 bias=True, 
                 feat_for_weight='xiea|xjea',
                 feat_for_aggr='xj',
                 residual=False,
                 **kwargs):
        '''
        Parameters:
            - node_feat_dim (int): node feature dimension.
            - edge_feat_dim (int): edge feature dimension.
            - hidd_feat_dim (int): hidden feature diemsion for weight calculation.
            - num_heads (int).
            - acti (str): acti function. Default = 'silu'
            - norm (str or None): norm for final aggregated features x_aggr and edge attention. Default = 'batch'
            - bias (bool): whether to add a bias vector to the final aggregated features x_aggr. Default = True
            - feat_for_weight (str): features that choose to weight and calculate attention. Default 'xiea|xjea'
              You can choose from 'xi|xj', 'xiea|xjea', 'xi|xj|ea'. | means weight seperately and then concatenate. 
              ea is edge_attr. xiea here means concatenate x_i and edge_attr at first and then weight. 
            - feat_for_aggr (str): features that use to multiplies with attention and then aggregates. Default 'xj'
              A linear layer is always use to transfrom its dimension to x.shape. Choose from 'xj', 'xi|xj', 'xi|xj|ea', | means concatenate.
              If you use 'xiea|xjea' to weight, when aggregate xi == xiea, xj == xjea, so you can not choose 'xi|xj|ea' at this time.
            - residual (bool): add the unaggregated feature x to the aggregated feature x_aggr as the output. Default False
        '''
        super().__init__(aggr='add', flow='target_to_source', **kwargs)
        self.model_args = {k: v for k, v in locals().items() if k not in ['self', '__class__', 'kwargs']}
        self.model_args.update(kwargs)
        
        self.node_feat_dim          = node_feat_dim 
        self.hidd_feat_dim          = hidd_feat_dim
        self.num_heads              = num_heads
        self.norm                   = norm
        self.feat_for_weight        = feat_for_weight
        self.feat_for_aggr          = feat_for_aggr
        self.residual               = residual
        # weight 
        if   feat_for_weight == 'xj': 
            in_feat_dim_weight      = node_feat_dim
        elif feat_for_weight == 'xi|xj': 
            in_feat_dim_weight      = node_feat_dim * 2
        elif feat_for_weight == 'xj|ea':
            in_feat_dim_weight      = node_feat_dim + edge_feat_dim
        elif feat_for_weight == 'xi|xj|ea':
            in_feat_dim_weight      = node_feat_dim * 2 + edge_feat_dim
        self.to_weight_matrix       = fully_connected_layer([in_feat_dim_weight, num_heads * hidd_feat_dim], acti=acti, norm=norm)
        # message
        if   feat_for_aggr == 'xj':
            in_feat_dim_message     = node_feat_dim
        elif feat_for_aggr == 'xi|xj':
            in_feat_dim_message     = node_feat_dim * 2
        elif feat_for_aggr == 'xj|ea':
            in_feat_dim_message     = node_feat_dim + edge_feat_dim
        elif feat_for_aggr == 'xi|xj|ea':
            in_feat_dim_message     = node_feat_dim * 2 + edge_feat_dim
        self.get_message            = fully_connected_layer([in_feat_dim_message, num_heads * node_feat_dim], acti=acti, norm=norm)
        
        self.xupdate_norm           = get_normalization(norm, hidd_feat_dim)
                    
        self.edge_attention = None
    
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, edge_index_i, x_i, x_j, edge_attr):
        # cat feature and weight 
        if   self.feat_for_weight == 'xj': 
            feat2weight = x_j
        elif self.feat_for_weight == 'xi|xj': 
            feat2weight = torch.cat([x_i, x_j], dim=-1)
        elif self.feat_for_weight == 'xj|ea':
            feat2weight = torch.cat([x_j, edge_attr], dim=-1)
        elif self.feat_for_weight == 'xi|xj|ea':
            feat2weight = torch.cat([x_i, x_j, edge_attr], dim=-1)
        feat2weight = self.to_weight_matrix(feat2weight).view(-1, self.num_heads, self.hidd_feat_dim).sum(dim=-1)
        # calculate and store attention scores
        edge_attention = graph_softmax(feat2weight, edge_index_i)
        self.edge_attention = edge_attention#.mean(dim=-1)
        # get message feature through fcl
        if   self.feat_for_aggr == 'xj':
            message_matrix = self.get_message(x_j)
        elif self.feat_for_aggr == 'xi|xj':
            message_matrix = self.get_message(torch.cat([x_i, x_j], dim=-1))
        elif self.feat_for_aggr == 'xj|ea':
            message_matrix = self.get_message(torch.cat([x_j, edge_attr], dim=-1))
        elif self.feat_for_aggr == 'xi|xj|ea':
            message_matrix = self.get_message(torch.cat([x_i, x_j, edge_attr], dim=-1))
        # mul message matrix and attention score
        x_message = (message_matrix.view(-1, self.num_heads, self.node_feat_dim) * edge_attention.view(-1, self.num_heads, 1)).transpose(0, 1)
        
        return x_message
    
    def update(self, aggr_out, x):
        x_update = aggr_out.mean(dim=0)
        if self.residual:
            x_update = x_update + x
        x_update = x_update if self.norm is None else self.xupdate_norm(x_update)
        return x_update
    
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
        
    def set_requires_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = requires_grad


class multi_head_attention(nn.Module):
    '''
    multi head attention layer with query, key and value matrices
    '''
    def __init__(
        self, 
        init_dim_query, 
        init_dim_key, 
        init_dim_value,
        final_dim_value,
        num_heads=2, 
        extend_head=False,
        layer_norm=True,
        residual_att=False,
        residual_out=False,
        acti='tanh',
        bias=True):
        '''
        Parameters:
            - init_dim_query (int): initial query feature dimension.
            - init_dim_key (int): initial key feature dimension.
            - init_dim_value (int): initial value feature dimension.
            - final_dim_value (int): output value feature dimension.
            - num_heads (int): number of attension heads.
            - extend_head (bool): the way to create multi-head attention matrices. Default = True
              If True, feature will extend by a factor of heads and return to original dimension by linear transformation. 
              If False, split feature into heads parts and concatenate to original dimension. final_dim_value/num_heads should be int.
            - layer_norm (bool): add layer normalization. Default = True
            - residual_att (bool): add query matrix to attension output value. Default = True
            - residual_out (bool): use a linear transfrom of attension output as a residual and add it to attension output. Default = True
            - acti (str): activation function. Default 'silu'
            - bias (bool): add bias for linear layers. Default = True
        '''
        super().__init__()
        self.model_args = {k: v for k, v in locals().items() if k not in ['self', '__class__']}
        
        self.num_heads              = num_heads
        self.extend_head            = extend_head
        self.layer_norm             = layer_norm
        self.residual_att           = residual_att
        self.residual_out           = residual_out
        
        if extend_head:
            self.dim_multy_head     = final_dim_value * num_heads
            self.dim_singl_head     = final_dim_value
            self.linear_m2shead     = nn.Linear(self.dim_multy_head, self.dim_singl_head, bias=bias)
        else:
            self.dim_multy_head     = final_dim_value
            self.dim_singl_head     = int(final_dim_value / num_heads)
            self.register_parameter('linear_m2shead', None)
            
        self.linear_query           = nn.Linear(init_dim_query, self.dim_multy_head, bias=bias)
        self.linear_key             = nn.Linear(init_dim_key,   self.dim_multy_head, bias=bias)
        self.linear_value           = nn.Linear(init_dim_value, self.dim_multy_head, bias=bias)
        
        self.acti_layer             = get_activation(acti)
        
        if self.layer_norm:
            self.norm_att           = get_normalization('layer', self.dim_singl_head)
            self.norm_out           = get_normalization('layer', final_dim_value)
        else:
            self.register_parameter('norm_att', None)
            self.register_parameter('norm_out', None)
        
        if self.residual_out:
            self.linear_residual    = nn.Linear(self.dim_multy_head, self.dim_multy_head, bias=bias)
        else:
            self.register_parameter('linear_residual', None)
            
        self.attention_weights      = None
              
    def forward(self, query, key, value, mask):
        query_ = self.linear_query(query)
        key_   = self.linear_key(key)
        value_ = self.linear_value(value)
        
        querys = torch.cat(query_.split(self.dim_singl_head, -1), 0)
        keys   = torch.cat(key_.split(self.dim_singl_head, -1), 0)
        values = torch.cat(value_.split(self.dim_singl_head, -1), 0)
        
        extend_mask = mask.unsqueeze(1)
        extend_mask = extend_mask.to(dtype=next(self.parameters()).dtype)
        extend_mask = (1.0 - extend_mask) * -1e9
        attention_mask = torch.cat([extend_mask for _ in range(self.num_heads)], 0)

        attention_score   = querys.bmm(keys.transpose(-1, -2)) / math.sqrt(self.dim_singl_head)
        attention_weights = F.softmax(attention_score + attention_mask, dim=-1)
        self.attention_weights = attention_weights.split(query.size(0), 0)
        
        attention_output  = attention_weights.bmm(values)
        if self.residual_att:
            attention_output = attention_output + querys
        if self.layer_norm:
            attention_output = self.norm_att(attention_output)
        attention_output = torch.cat(attention_output.split(query.size(0), 0), -1)
        
        if self.residual_out:
            attention_output = attention_output + self.acti_layer(self.linear_residual(attention_output))
        if self.extend_head:
            attention_output = self.linear_m2shead(attention_output)
            
        if self.layer_norm:
            attention_output = self.norm_out(attention_output)
        attention_output = self.acti_layer(attention_output)
        
        return attention_output
        
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
        
    def set_requires_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = requires_grad


class multi_head_attention_linear(nn.Module):
    '''
    The multi-head attention mechanism is applied to the vector, and the weight of each value in the vector is obtained through the linear layer
    '''
    def __init__(
    self, 
    feature_dim, 
    num_heads=3, 
    hidden_dim=[], 
    acti='tanh', 
    norm='batch'
    ):
        '''
        Parameters:
            - feature_dim (int).
            - num_heads (int). Default = 3
            - hidden_dim (list). Default = []
            - acti (str). Default = 'tanh'
            - norm (str or None). Default = 'batch'
        '''
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.att  = fully_connected_layer([feature_dim] + hidden_dim + [feature_dim * num_heads], acti=acti)
        self.norm = get_normalization(norm, feature_dim)
        self.acti = get_activation(acti)
        self.attention_scores = None
        
    def forward(self, x):
        attention_scores = F.softmax(self.att(x).view(-1, self.num_heads, self.feature_dim), dim=-1)
        self.attention_scores = attention_scores
        x = torch.sum(x.unsqueeze(1) * attention_scores, dim=1)
        
        if self.norm is not None:
            x = self.norm(x)
        x = self.acti(x)
        
        return x

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
        
    def set_requires_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = requires_grad


class multi_head_global_attention(nn.Module):
    '''
    Calculate multy head attention score for each node in a graph, and sum the node features weighted by the attention scores
    '''
    def __init__(
    self,
    feat_dim,
    hidd_dim=[],
    num_heads=3,
    acti='tanh',
    norm='batch'):
        '''
        Parameters:
            - feat_dim (int)
            - hidd_dim (list). Default = []
            - num_heads (int). Default = 3
            - acti (str). Default = 'tanh'
            - norm (str or None). Default = 'batch'
        '''
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.att = fully_connected_layer([feat_dim] + hidd_dim + [num_heads], acti=acti, norm=norm)
        self.attention_scores = None
        self.norm = get_normalization(norm, feat_dim)
        self.acti = get_activation(acti)
    
    def forward(self, x, batch):
        attention_scores = self.att(x).reshape(-1, self.num_heads, 1)
        attention_scores = graph_softmax(attention_scores, batch)
        self.attention_scores = attention_scores
        attention_scores = torch.repeat_interleave(attention_scores, repeats=self.feat_dim, dim=-1)
        x = torch.sum(x.unsqueeze(1) * attention_scores, dim=1)
        
        if self.norm is not None:
            x = self.norm(x)
        x = self.acti(x)

        return x
    
        
class weighted_add(nn.Module):
    '''
    Suggest features consist (cat) of several parts, calculate multy head attention scores for each part and sum each weighted part
    '''
    def __init__(self, feat_dim, class_num=3):
        '''
        Parameters:
            - feat_dim (int): total feature dimension
            - class_num (int): number of class (parts). Default = 3
        '''
        super().__init__()
        self.feat_dim = feat_dim
        self.class_num = class_num
        self.get_weight = fully_connected_layer([feat_dim * class_num, 3], acti='softmax')
        self.weight = None
        
    def forward(self, x):
        w = self.get_weight(x)
        self.weight = w
        x = x.view(-1, self.class_num, self.feat_dim)
        x = x * w.unsqueeze(-1)
        x = torch.sum(x, dim=1)
        
        return x
        
        
class fully_connected_layer(nn.Module):
    '''
    Fully connected layer (Multi-Layer Perceptron) used for non-linear regression
    
    '''
    def __init__(
    self, 
    dim_sequence, 
    acti='silu', 
    norm='batch', 
    dropout_rate=0., 
    bias=True,
    predict_reg=False,
    predict_cla=False,
    ):
        '''
        Parameters:
            - dim_sequence (array-like): dimensions of each layer, at least two dimensions.
            - acti (str): acti function after each linear layer. Default = 'silu'
            - norm ('batch', 'layer' or None): norm layer before the final linear layer. Default = 'layer'
            - dropout_rate (float): dropout rate of the dropout layer at last. Default = 0.
            - bias (bool): whether to add bias to each linear layer. Default = True
            - predict_reg (bool): whether this FCL is used to predict regression target, if so, fianl liner bias will be False. Default = False
            - predict_cla (bool): whether this FCL is used to predict Classification target, if so, final linear will go through norm and softmax. Default = False
        '''
        super().__init__()
        self.model_args = {k: v for k, v in locals().items() if k not in ['self', '__class__']}

        self.layers = nn.Sequential()
        number_layer = len(dim_sequence)
        
        for n in range(number_layer - 1):
            if   predict_reg and n == number_layer - 2:
                self.layers.append(nn.Linear(dim_sequence[n], dim_sequence[n + 1], bias=False))
            elif predict_cla and n == number_layer - 2:
                self.layers.append(nn.Linear(dim_sequence[n], dim_sequence[n + 1], bias=False))
                self.layers.append(get_activation('softmax'))
            else:
                self.layers.append(nn.Linear(dim_sequence[n], dim_sequence[n + 1], bias=bias))
                if norm is not None:
                    self.layers.append(get_normalization(norm, dim_sequence[n + 1]))
                if acti is not None:
                    self.layers.append(get_activation(acti))
                self.layers.append(nn.Dropout(dropout_rate))
            #elif (predict_reg or predict_cla) and n == number_layer - 3:
                #self.layers.append(nn.Linear(dim_sequence[n], dim_sequence[n + 1], bias=bias))
                #if norm is not None:
                    #self.layers.append(get_normalization(norm, dim_sequence[n + 1]))
                #if acti is not None:
                    #self.layers.append(get_activation(acti))
                #self.layers.append(nn.Dropout(dropout_rate))                    
            #elif n == number_layer - 2:
                #self.layers.append(nn.Linear(dim_sequence[n], dim_sequence[n + 1], bias=bias))
                #if norm is not None:
                    #self.layers.append(get_normalization(norm, dim_sequence[n + 1]))
                #if acti is not None:
                    #self.layers.append(get_activation(acti))
            #else:
                #self.layers.append(nn.Linear(dim_sequence[n], dim_sequence[n + 1], bias=bias))
            #if ((predict_reg or predict_cla) and n == number_layer - 4) or (not (predict_reg or predict_cla) and n == number_layer - 3):
                #self.layers.append(nn.Dropout(dropout_rate))
                        
    def forward(self, x):
        return self.layers(x)
    
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
        
    def set_requires_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = requires_grad
            
            
def cluster_pool(
    x, 
    cluster,
    batch,
    size=3,
    reduce='mean',
    batch_cluster=True,
):
    '''
    Pools node features according to the clustering defined in`cluster
    
    Parameters:
        - x (Tensor): the node feature matrix.
        - cluster (long Tensor): The cluster vector which assigns each node to a specific cluster.
        - batch (Tensor): batch tensor in graph Data
        - size (int):  number of cluster in graphs. Default = 3
        - reduce (str): the reduce operation, like 'sum', 'mean', 'max', 'min', 'mul'. Default = 'mean'
        - batch_cluster (bool): turn cluster index according to batch tensor. Default = True
    '''
    cluster = cluster.long()
    batch_size = int(batch.max().item()) + 1
    if batch_cluster:
        cluster = cluster + batch * size
        
    x_pool = scatter(x, cluster, dim=0, dim_size=size * batch_size, reduce=reduce)
    x_pool = x_pool.reshape(-1, x.size(-1) * size)
    
    return x_pool
    
    
def get_normalization(name, feat_dim):
    '''
    Return an normalization function using name.
    
    Parameters:
        - name (str, 'batch' or 'layer'): normalization layer name.
        - feat_dim (int): feature dimension.
    Return:
        - The normalization function
    '''
    if name is not None:
        return {
            'batch': nn.BatchNorm1d(feat_dim),
            'layer': nn.LayerNorm(feat_dim),
        }.get(name.lower(), None)
    else:
        return None


def get_activation(name):
    '''
    Return an activation function using the name.
    
    Parameters:
        - name (str): activation function name
    Return:
        - The activation function
    '''
    try:
        return {
            'relu': nn.ReLU(),
            'silu': nn.SiLU(),
            'gelu': nn.GELU(),
            'softplus': nn.Softplus(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leakyrelu': nn.LeakyReLU(),
            'softmax': nn.Softmax(dim=-1)
        }[name.lower()]
    except KeyError as exc:
        raise NotImplementedError from exc