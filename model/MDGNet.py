import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.SRGBlock import SRGBlock
from layers.ASDBlock import ASDBlock
from layers.AxialRelationAttention import AxialRelationAttention
from layers.MGRBlock import MGRBlock


class Model(nn.Module):
    """
    MDGNet: Multi-Domain Graph-Spectral Modeling for Long-Term Multivariate Time Series Forecasting
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.configs = configs
        self.recurrence = getattr(configs, 'R', 1) 
        if self.recurrence < 1:
            raise ValueError("Recurrence parameter R for JAA / AxialRelationAttention must be at least 1.")
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy

        # Graph relational branch (SRGBlock)
        self.use_gcn = getattr(configs, 'num_gcn_channels', 0) > 0
        self.srg_block = None
        if self.use_gcn:
            gcn_configs = type('GCNConfigs', (object,), {})() 
            gcn_configs.seq_len = configs.num_gcn_channels # N (number of variables/nodes for GCN)
            gcn_configs.d_model = configs.d_model # E (feature dimension)
            gcn_configs.top_k = getattr(configs, 'top_k', 2) 
            gcn_configs.d_ff = getattr(configs, 'd_ff', 4 * configs.d_model)
            gcn_configs.n_heads = getattr(configs, 'n_heads', 8)
            gcn_configs.dropout = getattr(configs, 'dropout', 0.1)
            gcn_configs.conv_channel = getattr(configs, 'conv_channel', 64)
            gcn_configs.skip_channel = getattr(configs, 'skip_channel', 256)
            gcn_configs.gcn_depth = getattr(configs, 'gcn_depth', 2)
            gcn_configs.propalpha = getattr(configs, 'propalpha', 0.05)
            gcn_configs.node_dim = getattr(configs, 'node_dim', 40)
            gcn_configs.pred_len = configs.pred_len
            self.srg_block = SRGBlock(gcn_configs)

        # Spectral branch (ASDBlock)
        self.use_freq_for_bypass = getattr(configs, 'use_freq_for_bypass', False)
        self.asd_block = None
        if self.use_freq_for_bypass:
            freq_configs = type('FreqConfigs', (object,), {})()
            freq_configs.kernel_size = getattr(configs, 'freq_kernel_size', 25)
            freq_configs.n_fft = getattr(configs, 'n_fft', [48])
            freq_configs.dropout = getattr(configs, 'dropout', 0.1)
            freq_configs.kernel_num = getattr(configs, 'kernel_num', 16)
            freq_configs.individual_factor = getattr(configs, 'individual_factor', 7)
            freq_configs.mode = getattr(configs, 'mode', 'MK')
            num_gcn_channels_to_process_at_init = getattr(configs, 'num_gcn_channels', 0)
            actual_seq_len_for_freq = 0
            enc_in_for_freq = configs.d_model
            if not self.use_gcn or num_gcn_channels_to_process_at_init == 0:
                actual_seq_len_for_freq = configs.seq_len # N_embedded
            elif self.use_gcn and num_gcn_channels_to_process_at_init > 0 and num_gcn_channels_to_process_at_init < configs.seq_len:
                actual_seq_len_for_freq = configs.seq_len - num_gcn_channels_to_process_at_init # N_bypass
            if actual_seq_len_for_freq > 0:
                self.asd_block = ASDBlock(
                    freq_configs,
                    seq_len_for_freq=actual_seq_len_for_freq, # N_bypass
                    enc_in_for_freq=enc_in_for_freq # E (configs.d_model)
                )
            else:
                self.use_freq_for_bypass = False

        # Multi-Domain Gated Recurrent Block (MGRBlock)
        self.mgr_block = MGRBlock(input_dim=configs.d_model, hidden_dim=configs.d_model, kernel_size=(3, 3),
                                 num_layers=1,
                                 batch_first=True, bias=True, return_all_layers=False)

        # Transformer Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # Joint Axial Attention (JAA)
        self.jaa = AxialRelationAttention(in_dim=configs.d_model)
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        B, L_original, N_original = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  
        N_embedded = enc_out.shape[1]
        E_features = enc_out.shape[2]
        fused_features = enc_out

        if self.use_gcn:
            num_gcn_channels_to_process = self.configs.num_gcn_channels
            
            if num_gcn_channels_to_process > N_embedded:
                raise ValueError(f"num_gcn_channels ({num_gcn_channels_to_process}) cannot be greater than embedded channels ({N_embedded}). "
                                 f"Please check your --num_gcn_channels setting or DataEmbedding_inverted behavior.")
            
            if num_gcn_channels_to_process == N_embedded:
                graph_branch_input = enc_out # (B, N_embedded, E_features)
                graph_branch_output = self.srg_block(graph_branch_input) # (B, N_embedded, E_features)
                fused_features = graph_branch_output
            elif num_gcn_channels_to_process > 0 and num_gcn_channels_to_process < N_embedded:
                graph_branch_input = enc_out[:, :num_gcn_channels_to_process, :] # (B, num_gcn_channels_to_process, E_features)
                spectral_branch_input = enc_out[:, num_gcn_channels_to_process:, :] # (B, N_embedded - num_gcn_channels_to_process, E_features)

                graph_branch_output = self.srg_block(graph_branch_input) # (B, num_gcn_channels_to_process, E_features)
                if self.use_freq_for_bypass and self.asd_block is not None:
                    spectral_branch_output = self.asd_block(spectral_branch_input) # (B, N_bypass, E_features)
                    fused_features = torch.cat([graph_branch_output, spectral_branch_output], dim=1) # (B, N_embedded, E_features)
                else:
                    fused_features = torch.cat([graph_branch_output, spectral_branch_input], dim=1) # (B, N_embedded, E_features)
            else:
                if self.use_freq_for_bypass and self.asd_block is not None:
                    fused_features = self.asd_block(enc_out) # (B, N_embedded, E_features)
                else:
                    fused_features = enc_out
        else:
            if self.use_freq_for_bypass and self.asd_block is not None:
                fused_features = self.asd_block(enc_out) # (B, N_embedded, E_features)
            else:
                fused_features = enc_out

        # Multi-Domain Gated Recurrent Block (MGRBlock)
        mgr_input = fused_features.unsqueeze(-1).unsqueeze(-1)  # (B, N_embedded, E_features, 1, 1)
        mgr_out, _ = self.mgr_block(mgr_input)
        mgr_out = mgr_out[0].squeeze(-1).squeeze(-1)  # (B, N_embedded, E_features)
        
        # Transformer Encoder
        enc_out, attns = self.encoder(mgr_out, attn_mask=None)

        # Joint Axial Attention (JAA)
        jaa_input = enc_out.unsqueeze(-1).unsqueeze(-1)  # (B, N_embedded, E_features, 1, 1)
        for _ in range(self.recurrence):
            jaa_input = self.jaa(jaa_input)
        jaa_output = jaa_input.squeeze(-1).squeeze(-1)  # (B, N_embedded, E_features)

        if N_embedded != N_original:
            # print(f"Warning: Embedded channels ({N_embedded}) do not match original channels ({N_original}). Slicing output to match original.")
            dec_out = self.projector(jaa_output[:, :N_original, :]).permute(0, 2, 1) # (B, Pred_Len, N_original)
        else:
            dec_out = self.projector(jaa_output).permute(0, 2, 1)  # (B, Pred_Len, N_original)

        if self.use_norm:
            stdev_repeated = stdev.repeat(1, self.pred_len, 1) # (B, Pred_Len, N_original)
            means_repeated = means.repeat(1, self.pred_len, 1) # (B, Pred_Len, N_original)

            dec_out = dec_out * stdev_repeated
            dec_out = dec_out + means_repeated

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
