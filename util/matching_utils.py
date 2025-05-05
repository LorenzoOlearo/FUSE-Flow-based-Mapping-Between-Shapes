import torch
import torch.nn.functional as F


def euler_to_SO3(euler_angles, convention = ['X', 'Y', 'Z']):
    '''
    :param euler_angles: [n, 6]
    :param convention: order of axis
    :return:
    '''

    def _axis_angle_rotation(axis, angle):
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        one = torch.ones_like(angle)
        zero = torch.zeros_like(angle)
        if axis == "X":
            R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
        elif axis == "Y":
            R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
        elif axis == "Z":
            R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
        else:
            raise ValueError("letter must be either X, Y or Z.")
        return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]

    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])




from model.models import FourierFeatsEncoding
from model.models import Swish
import torch.nn as nn

class MLP_xyz(nn.Module):
    def __init__(self,
        channels = 3,
        hidden_size = 256,
        depth = 6,):
    #input_dim: int = 3, time_dim: int = 1, hidden_dim: int = 128, fourier_encoding: str = 'FF', fourier_dim: int = 0):
        super().__init__()
        self.input_dim = channels
        self.hidden_dim = hidden_size
        self.ff_module = FourierFeatsEncoding(in_dim=4, num_frequencies=6, include_input=True)
        self.fourier_dim = ((3) * 6 * 2) + (3)
        self.rff_module = nn.Identity()

        self.main = nn.Sequential(
            nn.Linear(self.fourier_dim, self.hidden_dim),
            Swish(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            Swish(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            Swish(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            Swish(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )
    

    def forward(self, x):
        sz = x.size()
        
        h = x
        
        h = self.rff_module(h)
        h = self.ff_module(h)
       
        output = self.main(h)
        output = output.reshape(*sz)
        
        return output

import torch
import torch.nn as nn

# defien the neural map module
class Non_Linear_Map(nn.Module):
    '''
    This class defines a model composed of a linear module and a nonlinear module 
    to estimate a Neural Adjoint Map.
    '''
    def __init__(self, input_dim=128, output_dim=None, depth=4, width=128, act=nn.ReLU(), bias=False, nonlinear_type="MLP"):
        super().__init__()

        # Define default output dimension if None
        if output_dim is None:
            output_dim = input_dim
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.width = width

        self.nonlinear_branch = self._build_mlp(input_dim, output_dim, width, depth, act, bias)
        
        # Apply small scaling to MLP output for initialization
        self.mlp_scale = 0.01

        # Initialize weights
        self._reset_parameters()

    def forward(self, x):
        '''
        Forward pass through both the linear and non-linear branches.
        '''
        verts = x[:, :self.input_dim]


        # Nonlinear part
        t = self.mlp_scale * self.nonlinear_branch(verts)

        # Combine linear and nonlinear components
        x_out = x+t

        return x_out.squeeze()

    def _reset_parameters(self):
        '''
        Initialize the model parameters using Xavier uniform distribution.
        '''
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _build_mlp(self, input_dim, output_dim, width, depth, act, bias):
        '''
        Build an MLP (multi-layer perceptron) module.
        '''
        layers = []
        prev_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(prev_dim, width, bias=bias))
            layers.append(act)  # Add activation after each layer
            prev_dim = width
        layers.append(nn.Linear(prev_dim, output_dim, bias=bias))
        return nn.Sequential(*layers)

