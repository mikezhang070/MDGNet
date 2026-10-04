import torch
from torch import nn
import torch.nn.functional as F

def complex_tanh(input: torch.Tensor) -> torch.Tensor:
    return torch.tanh(input.real).type(torch.complex64) + 1j * torch.tanh(input.imag).type(torch.complex64)

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: (B, L, C)
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class ResidualSpectralEncoder(nn.Module):
    """
    Residual Spectral Encoder (formerly seasonal_encoder)
    mode:
      - 'MK': Mixed Kernel (shared kernels across channels)
      - 'IK': Independent Kernel (channel-wise kernels)
      - 'AK': Adaptive Kernel (learnable mix of MK & IK via alpha \in [0,1])
    """
    def __init__(self, enc_in, kernel_num, individual, mode, seq_len, n_fft, dropout=0.05):
        super().__init__()
        self.enc = int(enc_in)
        self.mode = mode
        self.n_fft = int(n_fft)
        if not isinstance(self.n_fft, int):
            raise ValueError(f"n_fft must be an integer, but got {type(self.n_fft)}")

        self.hop_length = max(1, int(self.n_fft * 0.5))
        effective_seq_len = max(int(seq_len), self.n_fft)
        self.window = (effective_seq_len - self.n_fft) // self.hop_length + 1
        if self.window <= 0:
            self.window = 1

        self.window_len = int(self.n_fft / 2) + 1 

        if self.mode == 'MK': #SSEBlock
            self.wg1 = nn.Parameter(torch.rand(kernel_num, self.window_len, self.window, dtype=torch.cfloat))
            self.wc  = nn.Parameter(torch.rand(kernel_num, self.window_len, self.window, self.window, dtype=torch.cfloat))
            nn.init.xavier_normal_(self.wg1)
            nn.init.xavier_normal_(self.wc)

        elif self.mode == 'IK': #ISEBlock
            self.wc1 = nn.Parameter(torch.rand(self.enc, individual, dtype=torch.cfloat))
            self.wc2 = nn.Parameter(torch.rand(individual, self.window_len, self.window, self.window, dtype=torch.cfloat))
            nn.init.xavier_normal_(self.wc1)
            nn.init.xavier_normal_(self.wc2)

        elif self.mode == 'AK': #HSEBlock
            self.wg1_shared = nn.Parameter(torch.rand(kernel_num, self.window_len, self.window, dtype=torch.cfloat))
            self.wc_shared  = nn.Parameter(torch.rand(kernel_num, self.window_len, self.window, self.window, dtype=torch.cfloat))
            nn.init.xavier_normal_(self.wg1_shared)
            nn.init.xavier_normal_(self.wc_shared)

            self.wc1_ind = nn.Parameter(torch.rand(self.enc, individual, dtype=torch.cfloat))
            self.wc2_ind = nn.Parameter(torch.rand(individual, self.window_len, self.window, self.window, dtype=torch.cfloat))
            nn.init.xavier_normal_(self.wc1_ind)
            nn.init.xavier_normal_(self.wc2_ind)

            self.alpha_param = nn.Parameter(torch.tensor(0.0))
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Choose from ['MK','IK','AK'].")

        self.wf1 = nn.Parameter(torch.rand(self.window_len, self.window_len, dtype=torch.cfloat))
        self.bf1 = nn.Parameter(torch.rand(self.window_len, 1, dtype=torch.cfloat))
        self.wf2 = nn.Parameter(torch.rand(self.window_len, self.window_len, dtype=torch.cfloat))
        self.bf2 = nn.Parameter(torch.rand(self.window_len, 1, dtype=torch.cfloat))

        nn.init.xavier_normal_(self.wf1)
        nn.init.xavier_normal_(self.bf1)
        nn.init.xavier_normal_(self.wf2)
        nn.init.xavier_normal_(self.bf2)

        self.dropout = nn.Dropout(p=dropout)

    def _stft_fix(self, q):
        original_seq_len = q.shape[-1]
        if original_seq_len < self.n_fft:
            padding_needed = self.n_fft - original_seq_len
            q = F.pad(q, (0, padding_needed), 'constant', 0)

        xq_stft = torch.stft(
            q, n_fft=self.n_fft, return_complex=True,
            hop_length=self.hop_length, center=False
        )  # (B*E, M, W0)

        if xq_stft.shape[2] != self.window:
            if xq_stft.shape[2] < self.window:
                padding_frames = self.window - xq_stft.shape[2]
                xq_stft = F.pad(xq_stft, (0, padding_frames), 'constant', 0)
            else:
                xq_stft = xq_stft[:, :, :self.window]

        return xq_stft, original_seq_len

    def _mk_branch(self, xq_stft):
        g_real = torch.sigmoid(torch.abs(torch.einsum("bmn,kmn->bk", xq_stft, self.wg1_shared)))
        g = self.dropout(g_real).type(torch.cfloat)

        h = torch.einsum("bmn,kmnw->bkmw", xq_stft, self.wc_shared)  # (B*E,k,M,W)
        out_shared = torch.einsum("bkmw,bk->bmw", h, g)              # (B*E,M,W)
        return out_shared

    def _ik_branch(self, xq_stft, q, B):
        E = self.enc
        # reshape -> (B, C, M, W)
        xq_stft_reshaped = xq_stft.reshape(int(q.shape[0] / E), E, xq_stft.shape[1], xq_stft.shape[2])
        wc_combined = torch.einsum("ci,imnw->cmnw", self.wc1_ind, self.wc2_ind)  # (C, M, W, W)
        out_ind = torch.einsum("bfhw,fhwo->bfho", xq_stft_reshaped, wc_combined)  # (B, C, M, W)
        out_ind = out_ind.reshape(out_ind.shape[0] * out_ind.shape[1], out_ind.shape[2], out_ind.shape[3])  # (B*C,M,W)
        return out_ind

    def forward(self, q):
        xq_stft, original_seq_len = self._stft_fix(q)  # (B*E, M, W)

        if self.mode == 'MK':
            g_real = torch.sigmoid(torch.abs(torch.einsum("bmn,kmn->bk", xq_stft, self.wg1)))
            g = self.dropout(g_real).type(torch.cfloat)                        # (B*E,k)
            h = torch.einsum("bmn,kmnw->bkmw", xq_stft, self.wc)               # (B*E,k,M,W)
            out = torch.einsum("bkmw,bk->bmw", h, g)                           # (B*E,M,W)

        elif self.mode == 'IK':
            xq_stft_reshaped = xq_stft.reshape(int(q.shape[0] / self.enc), self.enc, xq_stft.shape[1], xq_stft.shape[2])
            wc_combined = torch.einsum("ci,imnw->cmnw", self.wc1, self.wc2)    # (C,M,W,W)
            out = torch.einsum("bfhw,fhwo->bfho", xq_stft_reshaped, wc_combined)  # (B,C,M,W)
            out = out.reshape(out.shape[0] * out.shape[1], out.shape[2], out.shape[3])  # (B*E,M,W)

        else:  # 'AK'
            B = int(q.shape[0] / self.enc)
            out_shared = self._mk_branch(xq_stft)
            out_ind = self._ik_branch(xq_stft, q, B)
            alpha = torch.sigmoid(self.alpha_param)
            out = alpha * out_shared + (1.0 - alpha) * out_ind  # (B*E,M,W)
        out_res = out
        out = torch.einsum("biw,io->bow", out, self.wf1) + self.bf1.repeat(1, out.shape[2])
        out = complex_tanh(out)
        out = out_res + out

        out_res = out
        out = torch.einsum("biw,io->bow", out, self.wf2) + self.bf2.repeat(1, out.shape[2])
        out = complex_tanh(out)
        out = out_res + out

        out = torch.istft(out, n_fft=self.n_fft, hop_length=self.hop_length, center=False, length=original_seq_len)
        out = self.dropout(out)
        return out


class SmoothSpectralEncoder(nn.Module):
    """
    Smooth Spectral Encoder (formerly trend_encoder)
    """
    def __init__(self, seq_len, n_fft, dropout=0.05):
        super().__init__()
        self.n_fft = int(n_fft)
        if not isinstance(self.n_fft, int):
            raise ValueError(f"n_fft must be an integer, but got {type(self.n_fft)}")

        self.hop_length = max(1, int(self.n_fft * 0.5))
        effective_seq_len = max(int(seq_len), self.n_fft)
        self.window = (effective_seq_len - self.n_fft) // self.hop_length + 1
        if self.window <= 0:
            self.window = 1

        self.window_len = int(1 * (int(self.n_fft / 2) + 1))  # M
        self.wc  = nn.Parameter(torch.rand(self.window_len, self.window, self.window, dtype=torch.cfloat))
        self.wf1 = nn.Parameter(torch.rand(self.window_len, self.window_len, dtype=torch.cfloat))
        self.bf1 = nn.Parameter(torch.rand(self.window_len, 1, dtype=torch.cfloat))

        self.dropout = nn.Dropout(p=dropout)
        nn.init.xavier_normal_(self.wc)
        nn.init.xavier_normal_(self.wf1)
        nn.init.xavier_normal_(self.bf1)

    def forward(self, q):
        # q: (B*E, L)
        original_seq_len = q.shape[-1]
        if original_seq_len < self.n_fft:
            padding_needed = self.n_fft - original_seq_len
            q = F.pad(q, (0, padding_needed), 'constant', 0)

        xq_stft = torch.stft(q, n_fft=self.n_fft, return_complex=True,
                             hop_length=self.hop_length, center=False)  # (B*E,M,W)
        if xq_stft.shape[2] != self.window:
            if xq_stft.shape[2] < self.window:
                padding_frames = self.window - xq_stft.shape[2]
                xq_stft = F.pad(xq_stft, (0, padding_frames), 'constant', 0)
            else:
                xq_stft = xq_stft[:, :, :self.window]

        h = torch.einsum("bhi,hio->bho", xq_stft, self.wc)  # (B*E,M,W)
        h_res = h
        h = torch.einsum("biw,io->bow", h, self.wf1) + self.bf1.repeat(1, h.shape[2])
        h = complex_tanh(h)
        out = h_res + h

        out = torch.istft(out, n_fft=self.n_fft, hop_length=self.hop_length, center=False, length=original_seq_len)
        out = self.dropout(out)
        return out


class SpectralResolutionEncoder(nn.Module):
    """
    Spectral Resolution Encoder (formerly Encoder in ASDBlock)
    Encodes residual and smooth components at a specific spectral resolution (n_fft).
    """
    def __init__(self, enc_in, seq_len=512, kernel_num=16, individual=7, mode='MK', n_fft=16, dropout=0.05):
        super().__init__()
        self.residual_spectral_encoder = ResidualSpectralEncoder(
            enc_in=enc_in, kernel_num=kernel_num, individual=individual,
            mode=mode, seq_len=seq_len, n_fft=n_fft, dropout=dropout
        )
        self.smooth_spectral_encoder = SmoothSpectralEncoder(seq_len=seq_len, n_fft=n_fft, dropout=dropout)

    def forward(self, q1, q2):
        residual_encoded = self.residual_spectral_encoder(q1)
        smooth_encoded = self.smooth_spectral_encoder(q2)
        return residual_encoded, smooth_encoded


class ASDBlock(nn.Module):
    """
    Adaptive Spectral Decomposition Block (ASDBlock)
    Spectral branch decomposing inputs into residual and smooth spectral components.
    """
    def __init__(self, configs, seq_len_for_freq, enc_in_for_freq):
        super().__init__()
        kernel_size = getattr(configs, "kernel_size", getattr(configs, "moving_avg", 25))
        self.decompsition = series_decomp(kernel_size)

        n_fft_list = getattr(configs, "n_fft", 16)
        if not isinstance(n_fft_list, (list, tuple)):
            n_fft_list = [n_fft_list]
        n_fft_list = [int(x) for x in n_fft_list]

        kernel_num = getattr(configs, "kernel_num", getattr(configs, "freq_kernel_num", 16))
        individual_factor = getattr(configs, "individual_factor", 7)
        mode = getattr(configs, "mode", getattr(configs, "freq_mode", "MK"))
        dropout = getattr(configs, "dropout", 0.05)

        self.encoder_list = nn.ModuleList()
        for i_n_fft in n_fft_list:
            self.encoder_list.append(
                SpectralResolutionEncoder(
                    enc_in=enc_in_for_freq,
                    seq_len=seq_len_for_freq,
                    n_fft=i_n_fft,
                    dropout=dropout,
                    kernel_num=kernel_num,
                    individual=individual_factor,
                    mode=mode
                )
            )

        self.mlp1 = nn.Linear(len(n_fft_list), 1, bias=False)
        self.mlp2 = nn.Linear(len(n_fft_list), 1, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        x: (B, N_bypass, E)
        return: (B, N_bypass, E)
        """
        B, N_bypass, E = x.shape
        # (B, E, N_bypass)
        x_freq_input = x.permute(0, 2, 1)
        residual_component, smooth_component = self.decompsition(x_freq_input)  # (B,E,N)

        residual_component_reshaped = residual_component.reshape(B * E, N_bypass)
        smooth_component_reshaped = smooth_component.reshape(B * E, N_bypass)

        num_encoders = len(self.encoder_list)
        device = x.device
        out_residual_agg = torch.zeros((B * E, N_bypass, num_encoders), device=device, dtype=residual_component.dtype)
        out_smooth_agg   = torch.zeros((B * E, N_bypass, num_encoders), device=device, dtype=smooth_component.dtype)

        for idx, encoder in enumerate(self.encoder_list):
            r, s = encoder(residual_component_reshaped, smooth_component_reshaped)  # (B*E, N), (B*E, N)
            out_residual_agg[:, :, idx] = r
            out_smooth_agg[:, :, idx] = s

        if num_encoders > 1:
            out_residual = self.mlp1(out_residual_agg).squeeze(dim=-1)  # (B*E, N)
            out_smooth   = self.mlp2(out_smooth_agg).squeeze(dim=-1)    # (B*E, N)
        else:
            out_residual = out_residual_agg.squeeze(dim=-1)
            out_smooth   = out_smooth_agg.squeeze(dim=-1)

        out = out_residual + out_smooth  # (B*E, N)
        out = self.dropout(out)

        out = out.reshape(B, E, N_bypass).permute(0, 2, 1)
        return out

