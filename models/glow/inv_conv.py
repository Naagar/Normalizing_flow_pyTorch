import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class InvConv(nn.Module):
    """Invertible 1x1 Convolution for 2D inputs. Originally described in Glow
    (https://arxiv.org/abs/1807.03039). Does not support LU-decomposed version.

    Args:
        num_channels (int): Number of channels in the input and output.
    """
    def __init__(self, num_channels):
        super(InvConv, self).__init__()
        self.num_channels = num_channels

        # Initialize with a random orthogonal matrix
        w_init = np.random.randn(num_channels, num_channels)
        w_init = np.linalg.qr(w_init)[0].astype(np.float32)
        self.weight = nn.Parameter(torch.from_numpy(w_init))

    def forward(self, x, sldj, reverse=False):
        ldj = torch.slogdet(self.weight)[1] * x.size(2) * x.size(3)

        if reverse:
            weight = torch.inverse(self.weight.double()).float()
            sldj = sldj - ldj
        else:
            weight = self.weight
            sldj = sldj + ldj

        weight = weight.view(self.num_channels, self.num_channels, 1, 1)
        z = F.conv2d(x, weight)

        return z, sldj
def get_conv_square_ar_mask(h, w, n_in, n_out, zerodiagonal=False):
    """
    Function to get autoregressive convolution with square shape.
    """
    l = (h - 1) // 2
    m = (w - 1) // 2
    mask = np.ones([h, w, n_in, n_out], dtype=np.float32)
    mask[:l, :, :, :] = 0
    mask[:, :m, :, :] = 0
    mask[l, m, :, :] = get_linear_ar_mask(n_in, n_out, zerodiagonal)
    return mask

def get_conv_weight_np(filter_shape, stable_init=True, unit_testing=False):
    weight_np = np.random.randn(*filter_shape) * 0.02
    kcent = (filter_shape[0] - 1) // 2
    if stable_init or unit_testing:
        weight_np[kcent, kcent, :, :] += 1. * np.eye(filter_shape[3])
    weight_np = weight_np.astype('float32')
    return weight_np

class InvConv_3x3(nn.Module):
    """Invertible 3x3 Convolution for 2D inputs for CInC Flow. Originally described in Glow
    (https://arxiv.org/abs/1807.03039). Does not support LU-decomposed version.

    Args:
        num_channels (int): Number of channels in the input and output.
    """
    def __init__(self, num_channels):
        super(InvConv_3x3, self).__init__()
        self.num_channels = num_channels

        # Initialize with a random orthogonal matrix
        k_size = 2 * 3 - 1 
        k_cent  = (k_size -1) / 2
        mask_np = get_conv_square_ar_mask(k_size, k_size, self.num_channels, self.num_channels)
        mask_upsidedown_np = mask_np[::-1, ::-1, ::-1, ::-1].copy()
        mask_upsidedown = torch.tensor(mask_upsidedown_np)
        filter_shape = [k_size, k_size, self.num_channels, self.num_channels]
        
        w_init = np.random.randn(num_channels, num_channels, 3, 3)
        w_init = np.linalg.qr(w_init)[0].astype(np.float32)
        w_init = w_init * mask_upsidedown
        self.weight = nn.Parameter(torch.from_numpy(w_init))

    def forward(self, x, sldj, reverse=False):
        ldj = torch.slogdet(self.weight)[1] * x.size(2) * x.size(3)

        if reverse:
            weight = torch.inverse(self.weight.double()).float()
            sldj = sldj - ldj
        else:
            weight = self.weight
            sldj = sldj + ldj

        weight = weight.view(self.num_channels, self.num_channels, 1, 1)
        z = F.conv2d(x, weight)

        return z, sldj
