from math import sqrt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn, Tensor
from einops import rearrange
from einops.layers.torch import Rearrange
from utils.masking import TriangularCausalMask

class Predict(nn.Module):
    def __init__(self,  individual, c_out, seq_len, pred_len, dropout):
        super(Predict, self).__init__()
        self.individual = individual
        self.c_out = c_out

        if self.individual:
            self.seq2pred = nn.ModuleList()
            self.dropout = nn.ModuleList()
            for i in range(self.c_out):
                self.seq2pred.append(nn.Linear(seq_len , pred_len))
                self.dropout.append(nn.Dropout(dropout))
        else:
            self.seq2pred = nn.Linear(seq_len , pred_len)
            self.dropout = nn.Dropout(dropout)

    #(B,  c_out , seq)
    def forward(self, x):
        if self.individual:
            out = []
            for i in range(self.c_out):
                per_out = self.seq2pred[i](x[:,i,:])
                per_out = self.dropout[i](per_out)
                out.append(per_out)
            out = torch.stack(out,dim=1)
        else:
            out = self.seq2pred(x)
            out = self.dropout(out)

        return out

class Attention_Block(nn.Module):
    def __init__(self,  d_model, d_ff=None, n_heads=8, dropout=0.1, activation="relu"):
        super(Attention_Block, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = self_attention(FullAttention, d_model, n_heads=n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class self_attention(nn.Module):
    def __init__(self, attention, d_model ,n_heads):
        super(self_attention, self).__init__()
        d_keys =  d_model // n_heads
        d_values = d_model // n_heads

        self.inner_attention = attention( attention_dropout = 0.1)  # FullAttention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads


    def forward(self, queries ,keys ,values, attn_mask= None):
        B, L, _ = queries.shape   # [(B*scale_num), scale, d_model]
        _, S, _ = keys.shape  # B = B*scale_num, L = S = scale
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)  # [(B*scale_num), scale, d_model]->[(B*scale_num), scale, d_keys*n_heads]->[(B*scale_num), scale, n_heads, d_keys]
        keys = self.key_projection(keys).view(B, S, H, -1)  # [(B*scale_num), scale, n_heads, d_keys]
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
                    queries,
                    keys,
                    values,
                    attn_mask
                )  # [(B*scale_num), scale, n_heads,d_keys]
        out = out.view(B, L, -1)  # [(B*scale_num), scale, n_heads*d_keys]
        out = self.out_projection(out)  # [(B*scale_num), scale, d_model]
        return out, attn


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape  # [(B*scale_num), scale, n_heads, d_keys]
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)  # [(B*scale_num), n_heads, scale, scale]
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)  # [(B*scale_num), scale, n_heads,d_keys]
        # return V.contiguous()
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class GraphBlock(nn.Module):
    # 重构GraphBlock的__init__参数，使其更明确
    def __init__(self, num_nodes, feature_dim, conv_channel, skip_channel,
                        gcn_depth, dropout, propalpha, node_dim):
        super(GraphBlock, self).__init__()
        
        self.num_nodes = num_nodes # N (number of variables)
        self.feature_dim = feature_dim # E (embedding dimension)

        self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, node_dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(node_dim, self.num_nodes), requires_grad=True)
        
        # start_conv: Input (B, 1, E, N) -> Output (B, conv_channel, N, N)
        # Kernel height: E - N + 1
        kernel_height = self.feature_dim - self.num_nodes + 1
        if kernel_height <= 0:
            kernel_height = max(1, self.feature_dim - self.num_nodes + 1)
            if kernel_height == 1 and self.feature_dim < self.num_nodes:
                 print(f"Warning: GraphBlock kernel_height is 1 due to feature_dim ({self.feature_dim}) < num_nodes ({self.num_nodes}). This might indicate a design mismatch.")
            elif kernel_height <= 0: # Should not happen with max(1, ...)
                raise ValueError(f"GraphBlock: kernel_height for start_conv must be > 0. "
                                 f"Calculated: feature_dim ({self.feature_dim}) - num_nodes ({self.num_nodes}) + 1 = {self.feature_dim - self.num_nodes + 1}. "
                                 f"Consider adjusting feature_dim or num_nodes.")

        self.start_conv = nn.Conv2d(1, conv_channel, (kernel_height, 1))
        
        self.gconv1 = mixprop(conv_channel, skip_channel, gcn_depth, dropout, propalpha)
        self.gelu = nn.GELU()
        
        # end_conv: Input (B, skip_channel, N, N) -> Output (B, N, N, 1) -> squeeze -> (B, N, N)
        # Output channels: N, Kernel width: N
        self.end_conv = nn.Conv2d(skip_channel, self.num_nodes, (1, self.num_nodes))
        
        # linear: Input (B, N, N) -> Output (B, N, E)
        # in_features: N, out_features: E
        self.linear = nn.Linear(self.num_nodes, self.feature_dim)
        self.norm = nn.LayerNorm(self.feature_dim)

    # x in (B, N_nodes, E_features)
    def forward(self, x):
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        
        out = x.unsqueeze(1).transpose(2, 3) # (B, 1, E_features, N_nodes)
        
        out = self.start_conv(out) # (B, conv_channel, N_nodes, N_nodes)
        
        out = self.gelu(self.gconv1(out , adp)) # (B, skip_channel, N_nodes, N_nodes)
        
        out = self.end_conv(out).squeeze() # (B, N_nodes, N_nodes)
        
        out = self.linear(out) # (B, N_nodes, E_features)
        
        return self.norm(x + out)


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncwl,vw->ncvl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)
    
class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho


class simpleVIT(nn.Module):
    def __init__(self, in_channels, emb_size, patch_size=2, depth=1, num_heads=4, dropout=0.1,init_weight =True):
        super(simpleVIT, self).__init__()
        self.emb_size = emb_size
        self.depth = depth
        self.to_patch = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, 2 * patch_size + 1, padding= patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.layers = nn.ModuleList([])
        for _ in range(self.depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, dropout),
                FeedForward(emb_size,  emb_size)
            ]))

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self,x):
        B , N ,_ ,P = x.shape
        x = self.to_patch(x)
        for  norm ,attn, ff in self.layers:
            x = attn(norm(x)) + x
            x = ff(x) + x

        x = x.transpose(1,2).reshape(B, self.emb_size ,-1, P)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class TemporalExternalAttn(nn.Module):
    def __init__(self, scale, S=256):
        super().__init__()

        self.mk = nn.Linear(scale, S, bias=False)
        self.mv = nn.Linear(S, scale, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, queries):

        attn = self.mk(queries)  # bs,n,S
        attn = self.softmax(attn)  # bs,n,S
        # attn = attn / torch.sum(attn, dim=2, keepdim=True)  # bs,n,S

        out = self.mv(attn)  # bs,n,d_model
        return out
    
def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class SRGBlock(nn.Module):
    """
    Scale-Guided Relational Graph Convolution Block (SRGBlock)
    Graph relational branch capturing multi-scale cross-variate relational structure.
    """
    def __init__(self, configs):
        super(SRGBlock, self).__init__()
        # self.num_gcn_nodes refers to the number of nodes (N) for GCN,
        # which is configs.seq_len (set to num_gcn_channels in Model.__init__)
        self.num_gcn_nodes = configs.seq_len 
        self.pred_len = configs.pred_len # Not directly used in ScaleGraphBlock, but might be in configs
        self.k = configs.top_k

        self.att0 = Attention_Block(configs.d_model, configs.d_ff,
                                   n_heads=configs.n_heads, dropout=configs.dropout, activation="gelu")
        self.norm = nn.LayerNorm(configs.d_model)
        self.gelu = nn.GELU()
        self.gconv = nn.ModuleList()
        for i in range(self.k):
            self.gconv.append(
                GraphBlock(num_nodes=self.num_gcn_nodes, # N (configs.num_gcn_channels)
                           feature_dim=configs.d_model, # E (configs.d_model)
                           conv_channel=configs.conv_channel, 
                           skip_channel=configs.skip_channel,
                           gcn_depth=configs.gcn_depth, 
                           dropout=configs.dropout, 
                           propalpha=configs.propalpha, 
                           node_dim=configs.node_dim))


    def forward(self, x):
        # x is (B, N_nodes_for_GCN, E_features)
        B, N_nodes_for_GCN, E_features = x.size()
        
        # FFT_for_Period expects (B, T, C), so (B, N_nodes_for_GCN, E_features) is fine.
        # It will find periods along the N_nodes_for_GCN dimension.
        scale_list, scale_weight = FFT_for_Period(x, self.k)
        res = []
        
        # Store original x for residual connection
        original_x_for_residual = x 

        for i in range(self.k):
            scale = scale_list[i]
            
            # Gconv
            # GraphBlock expects (B, N_nodes_for_GCN, E_features)
            x_gcn_out = self.gconv[i](original_x_for_residual) # Apply GCN to the original input for each scale
            
            # Padding for FFT-based attention (if needed)
            # The padding logic here is based on the 'num_gcn_nodes'.
            if (N_nodes_for_GCN) % scale != 0:
                length = (((N_nodes_for_GCN) // scale) + 1) * scale
                padding = torch.zeros([x_gcn_out.shape[0], (length - (N_nodes_for_GCN)), x_gcn_out.shape[2]]).to(x_gcn_out.device)
                out = torch.cat([x_gcn_out, padding], dim=1)
            else:
                length = N_nodes_for_GCN
                out = x_gcn_out
            # out is (B, length, E_features)
            
            out = out.reshape(B, length // scale, scale, E_features)
            
            # For Multi-attention
            out = out.reshape(-1 , scale , E_features)
            out = self.norm(self.att0(out)) # att0 expects (B', L', d_model) -> (-1, scale, E_features)
            out = self.gelu(out)
            out = out.reshape(B, -1 , scale , E_features).reshape(B ,-1 ,E_features)
            
            out = out[:, :N_nodes_for_GCN, :] # Crop back to original N_nodes_for_GCN length
            res.append(out)

        res = torch.stack(res, dim=-1) # (B, N_nodes_for_GCN, E_features, k)
        
        # Adaptive aggregation
        # scale_weight is (B, k)
        # N_nodes here is N_nodes_for_GCN, E_features here is E_features
        scale_weight = F.softmax(scale_weight, dim=1) # (B, k)
        scale_weight = scale_weight.unsqueeze(1).unsqueeze(1).repeat(1, N_nodes_for_GCN, E_features, 1) # (B, N_nodes_for_GCN, E_features, k)
        res = torch.sum(res * scale_weight, -1) # (B, N_nodes_for_GCN, E_features)
        
        # Residual connection
        res = res + original_x_for_residual # Add back to the input of ScaleGraphBlock
        return res
