import torch
import torch.nn as nn
import torch.nn.functional as F

from models.glow.act_norm import ActNorm


class Coupling_1(nn.Module):
    """Affine coupling layer originally used in Real NVP and described by Glow.

    Note: The official Glow implementation (https://github.com/openai/glow)
    uses a different affine coupling formulation than described in the paper.
    This implementation follows the paper and Real NVP.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate activation
            in NN.
    """
    def __init__(self, in_channels, mid_channels):
        super(Coupling_1, self).__init__()
        self.nn = NN(in_channels, mid_channels, 2 * in_channels)
        self.scale = nn.Parameter(torch.ones(in_channels, 1, 1))

    def forward(self, x, ldj, reverse=False):
        x_1, x_2, x_3, x_id = x.chunk(4, dim=1)  # x_1, x_2, x_3 will be the modified by the coupling Layer 

        st = self.nn(x_1)
        s, t = st[:, 0::2, ...], st[:, 1::2, ...]
        s = self.scale * torch.tanh(s)

        # Scale and translate
        if reverse:
            x_1 = x_1 * s.mul(-1).exp() - t
            ldj = ldj - s.flatten(1).sum(-1)
        else:
            x_1 = (x_1 + t) * s.exp()
            ldj = ldj + s.flatten(1).sum(-1)

        st += self.nn(x_2)
        s, t = st[:, 0::2, ...], st[:, 1::2, ...]
        s = self.scale * torch.tanh(s)
        # Scale and translate
        if reverse:
            x_2 = x_2 * s.mul(-1).exp() - t
            ldj = ldj - s.flatten(1).sum(-1)
        else:
            x_2 = (x_2 + t) * s.exp()
            ldj = ldj + s.flatten(1).sum(-1)


        st += self.nn(x_3)
        s, t = st[:, 0::2, ...], st[:, 1::2, ...]
        s = self.scale * torch.tanh(s)
        # Scale and translate
        if reverse:
            x_3 = x_3 * s.mul(-1).exp() - t
            ldj = ldj - s.flatten(1).sum(-1)
        else:
            x_3 = (x_3 + t) * s.exp()
            ldj = ldj + s.flatten(1).sum(-1)

        x = torch.cat((x_1, x_2, x_3, x_id), dim=1)

        return x, ldj


class NN(nn.Module):
    """Small convolutional network used to compute scale and translate factors.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the hidden activations.
        out_channels (int): Number of channels in the output.
        use_act_norm (bool): Use activation norm rather than batch norm.
    """
    def __init__(self, in_channels, mid_channels, out_channels,
                 use_act_norm=False):
        super(NN, self).__init__()
        norm_fn = ActNorm if use_act_norm else nn.BatchNorm2d

        self.in_norm = norm_fn(in_channels)
        self.in_conv = nn.Conv2d(in_channels, mid_channels,
                                 kernel_size=3, padding=1, bias=False)
        nn.init.normal_(self.in_conv.weight, 0., 0.05)

        self.mid_norm = norm_fn(mid_channels)
        self.mid_conv = nn.Conv2d(mid_channels, mid_channels,
                                  kernel_size=1, padding=0, bias=False)
        nn.init.normal_(self.mid_conv.weight, 0., 0.05)

        self.out_norm = norm_fn(mid_channels)
        self.out_conv = nn.Conv2d(mid_channels, out_channels,
                                  kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x):
        x = self.in_norm(x)
        x = F.relu(x)
        x = self.in_conv(x)

        x = self.mid_norm(x)
        x = F.relu(x)
        x = self.mid_conv(x)

        x = self.out_norm(x)
        x = F.relu(x)
        x = self.out_conv(x)

        return x


# a = torch.zeros(100, 20, 32, 32)
# model_1 = Coupling_1()


# elif hps.flow_coupling == 2:
#                 block_size = n_z // 4
#                 z1 = z[:, :, :, :block_size]
#                 z2 = z[:, :, :, block_size:2*block_size]
#                 z3 = z[:, :, :, 2*block_size:3*block_size]
#                 z4 = z[:, :, :, 3*block_size:]

#                 y1 = z1
#                 h = f("f3", z[:,:,:,:3*block_size], hps.width, 2 * block_size)
#                 shift = h[:, :, :, 0::2]
#                 # scale = tf.exp(h[:, :, :, 1::2])
#                 scale = tf.nn.sigmoid(h[:, :, :, 1::2] + 2.)
#                 logscale = tf.log_sigmoid(h[:, :, :, 1::2] + 2.)
#                 y4 = z4 + shift
#                 y4 *= scale
#                 logdet += tf.reduce_sum(logscale, axis=[1, 2, 3])

#                 h = f("f2", z[:,:,:,:2*block_size], hps.width, 2 * block_size)
#                 shift = h[:, :, :, 0::2]
#                 # scale = tf.exp(h[:, :, :, 1::2])
#                 scale = tf.nn.sigmoid(h[:, :, :, 1::2] + 2.)
#                 logscale = tf.log_sigmoid(h[:, :, :, 1::2] + 2.)
#                 y3 = z3 + shift
#                 y3 *= scale
#                 logdet += tf.reduce_sum(logscale, axis=[1, 2, 3])

#                 h = f("f1", z[:,:,:,:block_size], hps.width, 2 * block_size)
#                 shift = h[:, :, :, 0::2]
#                 # scale = tf.exp(h[:, :, :, 1::2])
#                 scale = tf.nn.sigmoid(h[:, :, :, 1::2] + 2.)
#                 logscale = tf.log_sigmoid(h[:, :, :, 1::2] + 2.)
#                 y2 = z2 + shift
#                 y2 *= scale
#                 logdet += tf.reduce_sum(logscale, axis=[1, 2, 3])

#                 z = tf.concat([y1, y2, y3, y4], 3)