import torch
from torch import nn
import numpy as np
import utils.general as utils
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import distributions as dist
from torch.autograd import grad
import utils.plots as plt
from torch import distributions as dist
import logging


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out

class Conv1dGrad(nn.Conv1d):
    def forward(self, input, input_grad, compute_grad=False, is_first=False):
        output = super().forward(input)
        if not compute_grad:
            return output, None

        if len(self.weight.size())>2:
            weights = self.weight.squeeze(2)
            if input_grad != None:
                input_grad = input_grad.permute(0, 2, 1, 3)
        output_grad = weights[:,:3] if is_first else weights.matmul(input_grad)
        return output , output_grad


class TanHGrad(nn.Tanh):
    def forward(self, input, input_grad, compute_grad=False):
        output = super().forward(input)
        if not compute_grad:
            return output, None
        output_grad = (1 - torch.tanh(input).pow(2)).unsqueeze(-1) * input_grad
        return output, output_grad


class SoftplusGrad(nn.Softplus):
    def forward(self, input, input_grad, compute_grad=False):
        output = super().forward(input)
        if not compute_grad:
            return output, None
        output_grad = torch.sigmoid(self.beta * input).unsqueeze(-1).permute(0,2,1,3) * input_grad #
        return output , output_grad


class DCDFFN_Encoder(nn.Module):
    ''' PointNet-based encoder network with feature fusion.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=256, dim=3, hidden_dim=256):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Conv1d(dim, 2*hidden_dim, kernel_size=1, padding=0)
        self.fc_0 = nn.Conv1d(2*hidden_dim, hidden_dim, kernel_size=1, padding=0)
        self.fc_1 = nn.Conv1d(2*hidden_dim, hidden_dim, kernel_size=1, padding=0)
        self.fc_2 = nn.Conv1d(3*hidden_dim, hidden_dim, kernel_size=1, padding=0)
        self.fc_3 = nn.Conv1d(4*hidden_dim, hidden_dim,kernel_size=1, padding=0)
        self.fc_mean = nn.Linear(hidden_dim, c_dim)
        self.fc_std = nn.Linear(hidden_dim, c_dim)
        torch.nn.init.constant_(self.fc_mean.weight,0)
        torch.nn.init.constant_(self.fc_mean.bias, 0)

        torch.nn.init.constant_(self.fc_std.weight, 0)
        torch.nn.init.constant_(self.fc_std.bias, -10)

        self.actvn = nn.ReLU()
        self.pool = nn.MaxPool1d(3, stride=1, padding=1)
        
        self.pool2 = maxpool

    def forward(self, p):

        net = self.fc_pos(p)
        net = self.fc_0(self.actvn(net))
        net1 = net 

        pooled1 = self.pool2(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled1], dim=1)
        net = self.fc_1(self.actvn(net))
        net2 = net 

        pooled2 = self.pool2(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net1, net2, pooled2], dim=1)
        net = self.fc_2(self.actvn(net))
        net3 = net 

        pooled3 = self.pool2(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net1, net2, net3, pooled3], dim=1)

        net = self.fc_3(self.actvn(net))
        net = net.permute(0,2,1)

        f_c4_net = self.pool2(net, dim=1)
        f_c4_mean = self.fc_mean(self.actvn(f_c4_net))
        f_c4_std = self.fc_std(self.actvn(f_c4_net))
        return  f_c4_mean,  f_c4_std 


class DCDFFN_Decoder(nn.Module):
    """
    DeepSDF-based Decoder with feature fusion. 
    """


    def __init__(
            self,
            latent_size,
            dims,
            dropout=None,
            dropout_prob=0.0,
            norm_layers=(),
            latent_in=(),
            weight_norm=False,
            activation=None,
            latent_dropout=False,
            xyz_dim=3,
            geometric_init=True,
            beta=100
    ):
        super().__init__()

        bias = 1.0
        self.latent_size = latent_size
        last_out_dim = 1
        dims = [latent_size + xyz_dim] + dims + [last_out_dim]
        self.d_in = latent_size + xyz_dim
        self.latent_in = latent_in
        self.num_layers = len(dims)
        self.in_dim = 0
        for l in range(0, self.num_layers - 1):
            if l + 1 in latent_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            if l == 0:
                Conv1d = Conv1dGrad(dims[l], out_dim, kernel_size=1, padding=0)
                self.in_dim = self.in_dim + out_dim
                
            elif l!=0 and l < self.num_layers - 2:
                Conv1d = Conv1dGrad(self.in_dim, out_dim, kernel_size=1, padding=0)
                self.in_dim = self.in_dim + out_dim
            else:
                Conv1d = Conv1dGrad(self.in_dim, out_dim, kernel_size=1, padding=0)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(Conv1d.weight, mean=np.sqrt(np.pi) / np.sqrt(self.in_dim), std=0.0001)
                    torch.nn.init.constant_(Conv1d.bias, -bias)
                else:
                    torch.nn.init.constant_(Conv1d.bias, 0.0)
                    torch.nn.init.normal_(Conv1d.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                Conv1d = nn.utils.weight_norm(Conv1d)

            setattr(self, "Conv1d" + str(l), Conv1d)

        self.softplus = SoftplusGrad(beta=beta)
        

    def forward(self, input, latent, compute_grad=False, cat_latent=True):
        '''
        :param input: [shape: (N x d_in)]
        :param compute_grad: True for computing the input gradient. default=False
        :return: x: [shape: (N x d_out)]
                 x_grad: input gradient if compute_grad=True [shape: (N x d_in x d_out)]
                         None if compute_grad=False
        '''

        if len(input.size())!= 3:
            input = input.unsqueeze(0)
        x = input
        
        input_con = latent.unsqueeze(1).repeat(1, input.shape[1], 1) if self.latent_size > 0 else input
       
        if self.latent_size > 0 and cat_latent:
            x = torch.cat([x, input_con], dim=-1) if len(x.shape) == 3 else torch.cat(
                [x, latent.repeat(input.shape[0], 1)], dim=-1)
           
        input_con = x
        to_cat = x
        x_grad = None
        dfeatures = None
        ax_grad = None
        for l in range(0, self.num_layers - 1):
            Conv1d = getattr(self, "Conv1d" + str(l))
            if l == 0:
                if len(x.size())==3:
                    x = x.permute(0, 2, 1)
                x, x_grad = Conv1d(x, x_grad, compute_grad, l == 0)
                if l==0:
                    dfeatures = x
                if l==0 and x_grad is not None and compute_grad:
                    ax_grad = x_grad
            if l!=0 and l < self.num_layers-2:
                dfeatures = dfeatures / np.sqrt(2)
                if compute_grad:
                    ax_grad = ax_grad / np.sqrt(2) 
                x, x_grad = Conv1d(dfeatures, ax_grad, compute_grad, l == 0)

            if l == self.num_layers-2:
                dfeatures = dfeatures / np.sqrt(2)
                if compute_grad:
                    ax_grad = ax_grad / np.sqrt(2) 
                x, x_grad = Conv1d(dfeatures, ax_grad, compute_grad, l==0)

            if l < self.num_layers - 2:
                x, x_grad = self.softplus(x, x_grad, compute_grad)
                if l==0:
                    dfeatures = x
                if l==0 and x_grad is not None and compute_grad:
                    ax_grad = x_grad.permute(0, 2, 1, 3) 
                if l > 0:
                    if len(x.shape) == 3:
                        
                        dfeatures = torch.cat([dfeatures, x], axis=1) 
                    if len(x.shape) == 2:
                        
                        dfeatures = torch.cat([dfeatures, x], axis=0) 
           
                if l > 0 and x_grad is not None and compute_grad:
                    x_grad = x_grad.permute(0, 2, 1, 3)
                    ax_grad = torch.cat([ax_grad, x_grad], axis=1) 

                x = x.permute(0, 2, 1)
        return x, x_grad, input_con


class Network(nn.Module):
    def __init__(self, conf, latent_size, auto_decoder):
        super().__init__()
        
        self.latent_size = latent_size
        self.with_normals = conf.get_bool('encoder.with_normals')
        encoder_input_size = 6 if self.with_normals else 3
        self.encoder = DCDFFN_Encoder(dim=encoder_input_size) if not auto_decoder and latent_size > 0 else None

        self.implicit_map = DCDFFN_Decoder(latent_size=latent_size, **conf.get_config('decoder_implicit'))

        self.predict_normals_on_surfce = conf.get_bool('predict_normals_on_surfce')
        
        logging.debug("""self.latent_size = {0},
                      self.with_normals = {1}
                      self.predict_normals_on_surfce = {2}
                      """.format(self.latent_size,
                                                            self.with_normals,
                                                            self.predict_normals_on_surfce))

    def forward(self, manifold_points, manifold_normals, sample_nonmnfld, latent,
                only_encoder_forward, only_decoder_forward,epoch=-1):
        output = {}

        if self.encoder is not None and not only_decoder_forward:
            encoder_input = torch.cat([manifold_points, manifold_normals],
                                      axis=-1) if self.with_normals else manifold_points
            encoder_input = encoder_input.permute(0, 2, 1)
            qlm4, qls4 = self.encoder(encoder_input)
            
            q_z4 = dist.Normal(qlm4, torch.exp(qls4))
            l4 = q_z4.rsample()
            latent = l4
            q_latent_mean =  qlm4 
            q_latent_std =  qls4
            q_z = dist.Normal(q_latent_mean, torch.exp(q_latent_std))
            latent = q_z.rsample()
            latent_reg = (q_latent_mean.abs().mean(dim=-1) + (q_latent_std + 1).abs().mean(dim=-1))
            output['latent_reg'] = latent_reg

            if only_encoder_forward:
                return latent, q_latent_mean, torch.exp(q_latent_std)
        else:
            if only_encoder_forward:
                return None, None, None

        if only_decoder_forward:
            return self.implicit_map(manifold_points, latent, False)[0]
        else:
            non_mnfld_pred, non_mnfld_pred_grad, _ = self.implicit_map(sample_nonmnfld, latent, True)
            

            output['non_mnfld_pred_grad'] = non_mnfld_pred_grad
            output['non_mnfld_pred'] = non_mnfld_pred

            if not latent is None:
                output['norm_square_latent'] = (latent**2).mean(-1)

            if self.predict_normals_on_surfce:
                _, grad_on_surface, _ = self.implicit_map(manifold_points, latent, True)
                output['grad_on_surface'] = grad_on_surface

            return output

    

