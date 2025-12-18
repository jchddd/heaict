import math
import torch
import torch.nn               as nn
import torch.nn.functional    as F
from torch                    import Tensor
from torch_geometric.utils    import softmax as graph_softmax
from torch_geometric.utils    import scatter
from torch_geometric.nn.conv  import MessagePassing
        

class crystal_graph_convolution(MessagePassing):
    '''
    Crystal graph convolution layer which mainly use a Gate MLP (sigmoid(x_i|x_j|ea) * acti_core(x_i|x_j|ea)) to aggregated neighbor information
    '''
    def __init__(self, 
                 hidd_feat_dim,
                 hidd_mlp_dim=[],
                 acti_core='softplus',
                 acti='silu',
                 norm='batch', 
                 weight_by_edge=False,
                 cat_edge=True,
                 residual=True,
                 **kwargs):
        '''
        Parameters:
            - hidd_feat_dim (int): hidden feature dimension.
            - hidd_mlp_dim (array-like): hidden layers when transform x_i|x_j|ea to shape of x_i. Default []
            - acti_core (str): activation function for Gate MLP. Default 'softplus'
            - acti (str): activation function of all linear layers. Default 'silu'
            - norm (str or None): normalization layer for final aggregated features x_aggr. Default 'batch'
            - weight_by_edge (bool): weight the message feature by element-wise multiplication with weights obtained by linear transfrom of edge_attr. Default False
            - residual (bool): add the unaggregated feature x to the aggregated feature x_aggr as the output. Default True
        '''
        super().__init__(aggr='add', flow='target_to_source', **kwargs)
        self.model_args = {k: v for k, v in locals().items() if k not in ['self', '__class__', 'kwargs']}
        self.model_args.update(kwargs)
        # param
        self.norm                   = norm
        self.weight_by_edge         = weight_by_edge
        self.cat_edge               = cat_edge
        self.residual               = residual
        # dimension and activational function for fcls
        if cat_edge:
            num_feat_cat = 3
        else:
            num_feat_cat = 2
        dim_sequence                = [hidd_feat_dim * num_feat_cat] + hidd_mlp_dim + [hidd_feat_dim]
        self.sqeuence_core          = fully_connected_layer(dim_sequence, acti_core, norm)
        self.sqeuence_gate          = fully_connected_layer(dim_sequence, 'sigmoid', norm)
        # weight from edge
        if self.weight_by_edge:
            self.weight_edge        = fully_connected_layer([hidd_feat_dim, hidd_feat_dim], 'sigmoid', norm)
        else:
            self.register_parameter('weight_edge', None)
        # normalization layer
        self.xupdate_norm           = get_normalization(norm, hidd_feat_dim)
        self.xupdate_acti           = get_activation(acti)
        
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        # cat x_i, x_j, and edge_attr to z
        if self.cat_edge:
            z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            z = torch.cat([x_i, x_j], dim=-1)
        # z go through core and gate fcls
        x_message = self.sqeuence_core(z) * self.sqeuence_gate(z)
        # weight by edge
        if self.weight_by_edge:
            weight = self.weight_edge(edge_attr)
            x_message = x_message * weight
            
        return x_message
    
    def update(self, aggr_out, x):
        x_update = aggr_out + x if self.residual else aggr_out
        x_update = self.xupdate_acti(x_update) if self.norm is None else self.xupdate_acti(self.xupdate_norm(x_update))
        
        return x_update


class graph_attention_convolution(MessagePassing):
    '''
    Graph attention convolution layer which calculate attentions for neighbors of each node by a weight matrix W and a weight vector a
    '''
    def __init__(self, 
                 in_feat_dim,
                 out_feat_dim,
                 num_heads=3, 
                 acti='silu', 
                 norm='batch', 
                 bias=True, 
                 feat_for_weight='xj|ea',
                 feat_for_aggr='xj',
                 residual=True,
                 cat_output=True,
                 **kwargs):
        '''
        Parameters:
            - in_feat_dim (int): input feature dimension.
            - out_feat_dim (int): output feature dimension.
            - num_heads (int).
            - acti (str): acti function. Default = 'silu'
            - norm (str or None): norm for final aggregated features x_aggr and edge attention. Default = 'batch'
            - bias (bool): whether to add a bias vector to the final aggregated features x_aggr. Default = True
            - feat_for_weight (str): features that choose to weight and calculate attention. Default 'xj|ea'
              You can choose from 'xj', 'xi|xj', 'xj|ea', 'xi|xj|ea'. | means concatenate. 
            - feat_for_aggr (str): features that use to multiplies with attention and then aggregates. Default 'xj'
              feat for weight and aggr can be chose from 'xj', 'xi|xj', 'xj|ea', 'xi|xj|ea'. | means concatenate.
            - residual (bool): add the unaggregated feature x to the aggregated feature x_aggr as the output. Default True
            - cat_output (bool): cat output from each heads or mean them. Default = True
        '''
        super().__init__(aggr='add', flow='target_to_source', **kwargs)
        self.model_args = {k: v for k, v in locals().items() if k not in ['self', '__class__', 'kwargs']}
        self.model_args.update(kwargs)

        self.out_feat_dim           = out_feat_dim
        self.num_heads              = num_heads
        self.norm                   = norm
        self.residual               = residual
        self.cat_output             = cat_output
        # weight feature
        self.feat_for_weight        = feat_for_weight
        if   feat_for_weight == 'xj': 
            in_feat_dim_weight      = in_feat_dim
        elif feat_for_weight == 'xi|xj': 
            in_feat_dim_weight      = in_feat_dim * 2
        elif feat_for_weight == 'xj|ea':
            in_feat_dim_weight      = in_feat_dim * 2
        elif feat_for_weight == 'xi|xj|ea':
            in_feat_dim_weight      = in_feat_dim * 3
        # to weight matrix
        self.to_weight_matrix       = fully_connected_layer([in_feat_dim_weight, num_heads], acti=None, norm=None) #'leakyrelu'
        # message feature
        self.feat_for_aggr          = feat_for_aggr
        if   feat_for_aggr == 'xj':
            in_feat_dim_message     = in_feat_dim
        elif feat_for_aggr == 'xi|xj':
            in_feat_dim_message     = in_feat_dim * 2
        elif feat_for_aggr == 'xj|ea':
            in_feat_dim_message     = in_feat_dim * 2
        elif feat_for_aggr == 'xi|xj|ea':
            in_feat_dim_message     = in_feat_dim * 3
        self.get_message            = fully_connected_layer([in_feat_dim_message, num_heads * out_feat_dim], acti=None, norm=None)
        # bias
        if bias and cat_output:
            self.bias = nn.Parameter(torch.Tensor(self.num_heads * out_feat_dim), requires_grad=True)
            nn.init.zeros_(self.bias)
        elif bias and not cat_output:
            self.bias = nn.Parameter(torch.Tensor(out_feat_dim), requires_grad=True)
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
        # x_update
        self.xupdate_norm           = get_normalization(norm, out_feat_dim * num_heads if cat_output else out_feat_dim)
        self.xupdate_acti           = get_activation(acti)
                    
        self.edge_attention = None
    
    def forward(self, x, edge_index, edge_attr):
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return x
    
    def message(self, edge_index_i, x_i, x_j, edge_attr):
        # cat feature for weight 
        if   self.feat_for_weight == 'xj': 
            feat2weight = x_j
        elif self.feat_for_weight == 'xi|xj': 
            feat2weight = torch.cat([x_i, x_j], dim=-1)
        elif self.feat_for_weight == 'xj|ea':
            feat2weight = torch.cat([x_j, edge_attr], dim=-1)
        elif self.feat_for_weight == 'xi|xj|ea':
            feat2weight = torch.cat([x_i, x_j, edge_attr], dim=-1)
        # get edge att
        feat2weight = self.to_weight_matrix(feat2weight)
        edge_attention = graph_softmax(feat2weight, edge_index_i)
        self.edge_attention = edge_attention
        # get message feature
        if   self.feat_for_aggr == 'xj':
            message_feat = self.get_message(x_j)
        elif self.feat_for_aggr == 'xi|xj':
            message_feat = self.get_message(torch.cat([x_i, x_j], dim=-1))
        elif self.feat_for_aggr == 'xj|ea':
            message_feat = self.get_message(torch.cat([x_j, edge_attr], dim=-1))
        elif self.feat_for_aggr == 'xi|xj|ea':
            message_feat = self.get_message(torch.cat([x_i, x_j, edge_attr], dim=-1))
        # mul message matrix and attention score
        x_message = message_feat.view(-1, self.num_heads, self.out_feat_dim) * edge_attention.view(-1, self.num_heads, 1)
        if self.cat_output:
            x_message = x_message.view(-1, self.num_heads * self.out_feat_dim)
        else:
            x_message = x_message.mean(dim=1)
        
        return x_message
    
    def update(self, aggr_out, x):
        if self.bias is not None:
            aggr_out += self.bias
        x_update = aggr_out + x if self.residual else aggr_out
        x_update = self.xupdate_acti(x_update) if self.norm is None else self.xupdate_acti(self.xupdate_norm(x_update))
        
        return x_update


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
        residual_att=False,
        acti='silu',
        norm='batch'
    ):
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
            - norm (str or None): normalization layer. Default = 'batch'
            - residual_att (bool): add query matrix to attension output value. Default = False
            - acti (str): acti function. Default = 'silu'
            - norm (str or None): norm for final aggregated features x_aggr and edge attention. Default = 'batch'
        '''
        super().__init__()
        self.model_args = {k: v for k, v in locals().items() if k not in ['self', '__class__']}
        
        self.num_heads              = num_heads
        self.extend_head            = extend_head
        self.residual_att           = residual_att
        self.final_dim_value        = final_dim_value
        
        if extend_head:
            self.dim_multy_head     = final_dim_value * num_heads
            self.dim_singl_head     = final_dim_value
            self.linear_concat      = nn.Linear(final_dim_value * num_heads, final_dim_value)
        else:
            self.dim_multy_head     = final_dim_value
            self.dim_singl_head     = int(final_dim_value / num_heads)
            
        self.linear_query           = nn.Linear(init_dim_query, self.dim_multy_head)
        self.linear_key             = nn.Linear(init_dim_key,   self.dim_multy_head)
        self.linear_value           = nn.Linear(init_dim_value, self.dim_multy_head)

        self.output_norm           = get_normalization(norm, final_dim_value)
        self.output_acti           = get_activation(acti)
                    
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
        attention_weights = F.softmax(attention_score, dim=-1) #  + attention_mask
        self.attention_weights = attention_weights.split(query.size(0), 0)
        
        attention_output  = attention_weights.bmm(values)
        if self.residual_att:
            attention_output = attention_output + querys
        if not self.extend_head:
            attention_output = torch.cat(attention_output.split(query.size(0), 0), -1)
        elif self.extend_head:
            attention_output = self.linear_concat(attention_output)

        attention_output = attention_output.reshape(-1, self.final_dim_value)[mask.reshape(-1)]
        attention_output = self.output_acti(self.output_norm(attention_output))
                            
        return attention_output


class multi_head_attention_linear(nn.Module):
    '''
    The multi-head attention mechanism is applied to the vector, and the weight of each value in the vector is obtained through the linear layer
    '''
    def __init__(
        self, 
        feature_dim, 
        num_heads=3, 
        hidden_dim=[], 
    ):
        '''
        Parameters:
            - feature_dim (int).
            - num_heads (int). Default = 3
            - hidden_dim (list). Default = []
        '''
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.att  = fully_connected_layer([feature_dim] + hidden_dim + [feature_dim * num_heads], acti=None, norm=None)
        self.attention_scores = None
        
    def forward(self, x):
        attention_scores = F.softmax(self.att(x).view(-1, self.num_heads, self.feature_dim), dim=-1)
        self.attention_scores = attention_scores
        x = torch.sum(x.unsqueeze(1) * attention_scores, dim=1)
                
        return x
        
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
        hidden_dim=[],
        num_heads=3,
    ):
        '''
        Parameters:
            - feat_dim (int)
            - hidden_dim (list). Default = []
            - num_heads (int). Default = 3
        '''
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.att = fully_connected_layer([feat_dim] + hidden_dim + [num_heads], acti=None, norm=None)
        self.attention_scores = None
    
    def forward(self, x, batch):
        attention_scores = self.att(x).reshape(-1, self.num_heads, 1)
        attention_scores = graph_softmax(attention_scores, batch)
        self.attention_scores = attention_scores
        attention_scores = torch.repeat_interleave(attention_scores, repeats=self.feat_dim, dim=-1)
        x = torch.sum(x.unsqueeze(1) * attention_scores, dim=1)
        
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
                # self.layers.append(get_activation('softmax'))
            else:
                self.layers.append(nn.Linear(dim_sequence[n], dim_sequence[n + 1], bias=bias))
                if norm is not None:
                    self.layers.append(get_normalization(norm, dim_sequence[n + 1]))
                if acti is not None:
                    self.layers.append(get_activation(acti))
                self.layers.append(nn.Dropout(dropout_rate))
                        
    def forward(self, x):
        return self.layers(x)
            
            
def cluster_pool(
    x, 
    cluster,
    batch,
    size=3,
    reduce='sum',
    batch_cluster=True,
):
    '''
    Pools node features according to the clustering defined in graph.cluster and cat them into a tensor
    
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