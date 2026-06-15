#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List

class XtuningLayer():
    def __init__(
        self, 
        r: int, 
        xtuning_alpha: int, 
        xtuning_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.xtuning_alpha = xtuning_alpha
        # Optional dropout
        if xtuning_dropout > 0.:
            self.xtuning_dropout = nn.Dropout(p=xtuning_dropout)
        else:
            self.xtuning_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Embedding(nn.Embedding, XtuningLayer):
    # xtuning implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        xtuning_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        XtuningLayer.__init__(self, r=r, xtuning_alpha=xtuning_alpha, xtuning_dropout=0,
                           merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.xtuning_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.xtuning_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.xtuning_C = nn.Parameter(self.weight.new_zeros((r, r)))
            self.scaling = self.xtuning_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            print("=====================================enter new Xtuning Embedding method========================================")
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'xtuning_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.xtuning_A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.xtuning_C, a=math.sqrt(5))
            nn.init.zeros_(self.xtuning_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.xtuning_B @ (self.xtuning_C @ self.xtuning_C.transpose(0, 1)) @ self.xtuning_A).transpose(0, 1) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.xtuning_B @ (self.xtuning_C @ self.xtuning_C.transpose(0, 1)) @ self.xtuning_A).transpose(0, 1) * self.scaling
                self.merged = True
        
    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result_l = nn.Embedding.forward(self, x)
            after_A = F.embedding(
                x, self.xtuning_A.transpose(0, 1), self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
            )
            result_r = (after_A @ (self.xtuning_C.transpose(0, 1) @ self.xtuning_C) @ self.xtuning_B.transpose(0, 1)) * self.scaling
            result = result_l + result_r
            return result
        else:
            return nn.Embedding.forward(self, x)
            

class Linear(nn.Linear, XtuningLayer):
    # Xtuning implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        xtuning_alpha: int = 1, 
        xtuning_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        XtuningLayer.__init__(self, r=r, xtuning_alpha=xtuning_alpha, xtuning_dropout=xtuning_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.xtuning_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.xtuning_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.xtuning_C = nn.Parameter(self.weight.new_zeros((r, r)))
            self.scaling = self.xtuning_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            print("=====================================enter new Xtuning Linear method==========CCCCCCCCCCCCCCCCCCCCCTTTTTTTTTTTTTTTTTTTTTTTTT==============================")
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'xtuning_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.xtuning_A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.xtuning_C, a=math.sqrt(5))
            nn.init.zeros_(self.xtuning_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.xtuning_B @ (self.xtuning_C @ self.xtuning_C.transpose(0, 1)) @ self.xtuning_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.xtuning_B @ (self.xtuning_C @ self.xtuning_C.transpose(0, 1)) @ self.xtuning_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result_l = F.linear(x, T(self.weight), bias=self.bias)
            result_r = (self.xtuning_dropout(x) @ self.xtuning_A.transpose(0, 1) @ (self.xtuning_C @ self.xtuning_C.transpose(0, 1)).transpose(0, 1) @ self.xtuning_B.transpose(0, 1)) * self.scaling
            result = result_l + result_r
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

class Linear4Module(nn.Module, XtuningLayer):
    # Xtuning implemented in a dense layer
    def __init__(
        self,
        nf: int,
        nx: int,
        r: int = 0,
        xtuning_alpha: int = 1,
        xtuning_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True
    ):
        super(Linear4Module, self).__init__()
        XtuningLayer.__init__(self, r=r, xtuning_alpha=xtuning_alpha, xtuning_dropout=xtuning_dropout,
                           merge_weights=merge_weights)
        self.nf = nf
        self.nx = nx
        self.fan_in_fan_out = fan_in_fan_out
        w = torch.empty(self.nx, self.nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))
        # Actual trainable parameters
        if r > 0:
            self.xtuning_A = nn.Parameter(self.weight.new_zeros((r, self.nx)))
            self.xtuning_B = nn.Parameter(self.weight.new_zeros((self.nf, r)))
            self.xtuning_C = nn.Parameter(self.weight.new_zeros((r, r)))
            self.scaling = self.xtuning_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            print("=====================================enter new Xtuning Linear4Module method=================CCCCCCCCCCCCCCCCCCCCCTTTTTTTTTTTTTTTTTTTTTTTTT=======================")
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        if hasattr(self, 'xtuning_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.xtuning_A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.xtuning_C, a=math.sqrt(5))
            nn.init.zeros_(self.xtuning_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.xtuning_B @ (self.xtuning_C @ self.xtuning_C.transpose(0, 1)) @ self.xtuning_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.xtuning_B @ (self.xtuning_C @ self.xtuning_C.transpose(0, 1)) @ self.xtuning_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        size_out = x.size()[:-1] + (self.nf,)
        x = x.view(-1, x.size(-1))
        if self.r > 0 and not self.merged:
            result_l = torch.addmm(self.bias, x, self.weight)
            result_r = (self.xtuning_dropout(x) @ self.xtuning_A.transpose(0, 1) @ (self.xtuning_C @ self.xtuning_C.transpose(0, 1)).transpose(0, 1) @ self.xtuning_B.transpose(0, 1)) * self.scaling
            result = result_l + result_r
            result = result.view(*size_out)
            return result
        else:
            result = torch.addmm(self.bias, x, self.weight)
            result = result.view(*size_out)
            return result

class MergedLinear(nn.Linear, XtuningLayer):
    # Xtuning implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        xtuning_alpha: int = 1, 
        xtuning_dropout: float = 0.,
        enable_xtuning: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        XtuningLayer.__init__(self, r=r, xtuning_alpha=xtuning_alpha, xtuning_dropout=xtuning_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_xtuning) == 0, \
            'The length of enable_xtuning must divide out_features'
        self.enable_xtuning = enable_xtuning
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_xtuning):
            self.xtuning_A = nn.Parameter(self.weight.new_zeros((r * sum(enable_xtuning), in_features)))
            self.xtuning_B = nn.Parameter(self.weight.new_zeros((out_features // len(enable_xtuning) * sum(enable_xtuning), r))) # weights for Conv1D with groups=sum(enable_xtuning)
            self.xtuning_C = nn.Parameter(self.weight.new_zeros((r * sum(enable_xtuning), r * sum(enable_xtuning))))
            self.scaling = self.xtuning_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.xtuning_ind = self.weight.new_zeros((out_features, ), dtype=torch.bool).view(len(enable_xtuning), -1)
            self.xtuning_ind[enable_xtuning, :] = True
            self.xtuning_ind = self.xtuning_ind.view(-1)
            print("=====================================enter new Xtuning MergedLinear method=================CCCCCCCCCCCCCCCCCCCCCTTTTTTTTTTTTTTTTTTTTTTTTT=======================")
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'xtuning_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.xtuning_A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.xtuning_C, a=math.sqrt(5))
            nn.init.zeros_(self.xtuning_B)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.xtuning_ind), *x.shape[1:]))
        result[self.xtuning_ind] = x
        return result

    def merge_ABC(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        c = self.xtuning_C @ self.xtuning_C.transpose(0, 1)
        pre_w = F.conv1d(
            self.xtuning_A.unsqueeze(0),
            c.unsqueeze(-1)
        ).squeeze(0)
        
        delta_w = F.conv1d(
            pre_w.unsqueeze(0),
            self.xtuning_B.unsqueeze(-1), 
            groups=sum(self.enable_xtuning)
        ).squeeze(0)
        return T(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_xtuning):
                    self.weight.data -= self.merge_ABC() * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_xtuning):
                    self.weight.data += self.merge_ABC() * self.scaling
                self.merged = True        

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result_l = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result_r = self.xtuning_dropout(x) @ T(self.merge_ABC().T) * self.scaling
                result = result_l + result_r
            return result

class Convxtuning(nn.Module, XtuningLayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=0, xtuning_alpha=1, xtuning_dropout=0., merge_weights=True, **kwargs):
        super(Convxtuning, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        XtuningLayer.__init__(self, r=r, xtuning_alpha=xtuning_alpha, xtuning_dropout=xtuning_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.xtuning_A = nn.Parameter(self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size)))
            self.xtuning_B = nn.Parameter(self.conv.weight.new_zeros((out_channels//self.conv.groups*kernel_size, r*kernel_size)))
            self.xtuning_C = nn.Parameter(self.conv.weight.new_zeros((r * kernel_size, r * r)))
            self.scaling = self.xtuning_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.conv.weight.requires_grad = False
            print("=====================================enter new Xtuning Convxtuning method========================================")
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'xtuning_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.xtuning_A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.xtuning_C, a=math.sqrt(5))
            nn.init.zeros_(self.xtuning_B)

    def train(self, mode=True):
        super(Convxtuning, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.conv.weight.data -= (self.xtuning_B @ (self.xtuning_C @ self.xtuning_C.transpose(0, 1)) @ self.xtuning_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.conv.weight.data += (self.xtuning_B @ (self.xtuning_C @ self.xtuning_C.transpose(0, 1)) @ self.xtuning_A).view(self.conv.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(
                x, 
                self.conv.weight + (self.xtuning_B @ (self.xtuning_C @ self.xtuning_C.transpose(0, 1)) @ self.xtuning_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        return self.conv(x)

class Conv2d(Convxtuning):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)

class Conv1d(Convxtuning):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)

# Can Extend to other ones like this

class Conv3d(Convxtuning):
    def __init__(self, *args, **kwargs):
        super(Conv3d, self).__init__(nn.Conv3d, *args, **kwargs)
