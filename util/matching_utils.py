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




###############  matching functions   

import sys
sys.path.append('..')
import numpy as np

from util.plot import start_end_subplot

from model import models, networks
from model.models import EDMPrecond,FMCond
from model.networks import MLP,MLP_general
import trimesh
from sklearn.neighbors import NearestNeighbors
from util.mesh_utils import *

device='cuda:0'
torch.cuda.set_device(device)



def compute_p2p_with_geomdist(source_input,target_input, source_model, target_model):


    with torch.no_grad():
        emb1_pullback=source_model.inverse(samples=source_input,num_steps=64)
        sample = target_model.sample(noise=emb1_pullback, num_steps=64)
    
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(target_input.cpu().numpy())
    _, p2p = nbrs.kneighbors(sample.cpu().numpy())
    p2p=p2p[:,0]

    return p2p


def compute_p2p_with_knn_gauss(source_input,target_input, source_model, target_model):
    with torch.no_grad():
        emb1_pullback=source_model.inverse(samples=source_input,num_steps=64)
        emb2_pullback = target_model.inverse(samples= target_input,num_steps=64)
    
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(emb2_pullback.cpu().numpy())
    _, p2p = nbrs.kneighbors(emb1_pullback.cpu().numpy())
    p2p=p2p[:,0]

    return p2p


def compute_p2p_with_knn(source_input,target_input):
    
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(target_input.cpu().numpy())
    _, p2p = nbrs.kneighbors(source_input.cpu().numpy())
    p2p=p2p[:,0]

    return p2p

from pyFM.mesh import TriMesh
from pyFM.functional import FunctionalMapping


fit_params = {
    'w_descr': 1e0,
    'w_lap': 1e-2,
    'w_dcomm': 1e-1,
    'w_orient': 0
}

def compute_p2p_with_fmaps(source_path, target_path, lm_source, lm_target):
    mesh1= normalize_mesh(trimesh.load(source_path, process=False))
    mesh2= normalize_mesh(trimesh.load(target_path, process=False))

    mesh1= TriMesh(mesh1.vertices, mesh1.faces).process()
    mesh2= TriMesh(mesh2.vertices, mesh2.faces).process()
    
    
    process_params = {
        'n_ev': (20,20),  # Number of eigenvalues on source and Target
        'landmarks':np.concat([lm_target[None],lm_source[None]],0).T,  # Landmarks
        'subsample_step': 5,  # In order not to use too many descriptors
        'descr_type': 'WKS',  # WKS or HKS
    }
    model_fm = FunctionalMapping(mesh2,mesh1)
    model_fm.preprocess(**process_params,verbose=False)

    model_fm.fit(**fit_params, verbose=False)

    p2p = model_fm.get_p2p(n_jobs=1)
    
    return p2p