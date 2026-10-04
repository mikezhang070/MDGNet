import torch
import torch.nn as nn


class AxialRelationAttention(nn.Module):
    """
    Joint Axial Attention (JAA) / Axial Relation Attention Module
    Computes axial attention across horizontal and vertical dimensions.
    """
    @staticmethod
    def _get_inf_mask(B_times_T, H, W, device):
        return -torch.diag(torch.tensor(float("inf"), device=device).repeat(H), 0).unsqueeze(0).repeat(B_times_T * W, 1, 1)

    def __init__(self, in_dim):
        super(AxialRelationAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if x.dim() == 5:
            B, T, C, H, W = x.size()
            x = x.view(B * T, C, H, W)
        else:
            B, C, H, W = x.size()
            T = 1
        proj_query = self.query_conv(x) # (B*T, C//8, H, W)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(B * T * W, -1, H).permute(0, 2, 1) # (B*T*W, H, C//8)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(B * T * H, -1, W).permute(0, 2, 1) # (B*T*H, W, C//8)
        proj_key = self.key_conv(x) # (B*T, C//8, H, W)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(B * T * W, -1, H) # (B*T*W, C//8, H)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(B * T * H, -1, W) # (B*T*H, C//8, W)

        proj_value = self.value_conv(x) # (B*T, C, H, W)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(B * T * W, -1, H) # (B*T*W, C, H)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(B * T * H, -1, W) # (B*T*H, C, W)

        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self._get_inf_mask(B * T, H, W, x.device))
        energy_H = energy_H.view(B * T, W, H, H).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(B * T, H, W, W)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        att_H = concate[:, :, :, 0:H].permute(0, 2, 1, 3).contiguous().view(B * T * W, H, H)
        att_W = concate[:, :, :, H:H + W].contiguous().view(B * T * H, W, W)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(B * T, W, -1, H).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(B * T, H, -1, W).permute(0, 2, 1, 3)
        output = self.gamma * (out_H + out_W) + x
        if T > 1:
            output = output.view(B, T, C, H, W)

        return output

JAA = AxialRelationAttention
