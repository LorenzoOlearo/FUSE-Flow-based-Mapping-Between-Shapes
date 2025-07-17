# SDF Training and Evaluation Script
# This script is designed to train a neural network to learn the signed distance function (SDF) of a 3D mesh.
#
# This code is adapted from:
# Implicit-ARAP: Efficient Handle-Guided Deformation of High-Resolution Meshes and Neural Fields via Local Patch Meshing
# https://github.com/dbaieri/implicit-arap/tree/main
# Grazie Daniele :)

import os
import argparse
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Literal, Optional, Set, Tuple, Type, List
import torch.autograd as ad

import mcubes
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.autograd import grad
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

import kaolin as kal
import trimesh
from dataclasses import dataclass, field

min_coord: float = -1.0
max_coord: float =  1.0
resolution: int = 512
chunk: int = 65536
landmarks_indices: List[Int] = [412, 5891,6593,3323,2119]

h = (max_coord - min_coord) / (resolution - 1) # Voxel spacing
EPSILON = 2 * h


class PrintableConfig:
    """Printable Config defining str function"""

    def __str__(self):
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if isinstance(val, Tuple):
                flattened_val = "["
                for item in val:
                    flattened_val += str(item) + "\n"
                flattened_val = flattened_val.rstrip("\n")
                val = flattened_val + "]"
            lines += f"{key}: {str(val)}".split("\n")
        return "\n    ".join(lines)
    
    def to_dict(self):
        out = {}
        for key, val in vars(self).items():
            if key.startswith('_'): 
                continue
            if isinstance(val, PrintableConfig):
                out[key] = val.to_dict()
            else:
                out[key] = val
        return out


# Base instantiate configs
@dataclass
class InstantiateConfig(PrintableConfig):
    """Config class for instantiating an the class specified in the _target attribute."""

    _target: Type

    def setup(self, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        return self._target(self, **kwargs)
    

@dataclass
class FactoryConfig(PrintableConfig):
    """Config class for instantiating objects given their class name and module."""

    _name: str
    _module: ModuleType

    def setup(self, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        target = getattr(self._module, self._name)
        add_kwargs = vars(self).copy()
        add_kwargs.pop("_module")
        add_kwargs.pop("_name")
        add_kwargs.update(kwargs)
        return target(**add_kwargs)


class FourierFeatsEncoding(nn.Module):
    """Fourier feature encoding."""

    def __init__(self, in_dim: int, num_frequencies: int, include_input: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        # Frequencies: 2^0, ..., 2^{num_frequencies - 1}
        self.freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)

    def get_out_dim(self) -> int:
        return self.in_dim * (2 * self.num_frequencies + (1 if self.include_input else 0))

    def forward(self, x: Tensor) -> Tensor:
        out = []
        if self.include_input:
            out.append(x)
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=-1)


@dataclass
class MLPConfig:
    in_dim=3
    num_layers=8
    layer_width=256
    out_dim=1
    skip_connections=(4,)
    activation='Softplus'
    act_defaults={'beta': 100}
    num_frequencies=6
    encoding_with_input=True
    geometric_init=True
    out_activation: bool = False
    encoding_with_input: bool = True


class MLP(nn.Module):
    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        self.in_dim = config.in_dim
        assert self.in_dim > 0
        self.out_dim = config.out_dim if config.out_dim is not None else config.layer_width
        self.num_layers = config.num_layers
        self.layer_width = config.layer_width
        self.skip_connections = config.skip_connections
        self._skip_connections: Set[int] = set(self.skip_connections) if self.skip_connections else set()
        if 0 in self._skip_connections:
            self._skip_connections.remove(0)
        self.activation = getattr(nn, config.activation)(**config.act_defaults)
        self.out_activation = config.out_activation
        if config.num_frequencies > 0:
            self.encoding = FourierFeatsEncoding(self.in_dim, config.num_frequencies, config.encoding_with_input)
            self.mlp_in_dim = self.encoding.get_out_dim()
        else:
            self.encoding = None
            self.mlp_in_dim = self.in_dim

        self.build_nn_modules()
        if config.geometric_init:
            self.geometric_init()
            

    def build_nn_modules(self) -> None:
        layers = []
        if self.num_layers == 1:
            layers.append(nn.Linear(self.mlp_in_dim, self.out_dim))
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    assert i not in self._skip_connections
                    layers.append(nn.Linear(self.mlp_in_dim, self.layer_width))
                elif i in self._skip_connections:
                    layers.append(nn.Linear(self.layer_width + self.mlp_in_dim, self.layer_width))
                else:
                    layers.append(nn.Linear(self.layer_width, self.layer_width))
            layers.append(nn.Linear(self.layer_width, self.out_dim))
        self.layers = nn.ModuleList(layers)
        

    def geometric_init(self):
        for j, lin in enumerate(self.layers):
            if j == len(self.layers) - 1:
                if self.out_dim > 1:
                    torch.nn.init.zeros_(lin.weight)
                    torch.nn.init.zeros_(lin.bias)
                    return
                else:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(lin.in_features), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -0.5)
            elif self.encoding is not None and j == 0:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.constant_(lin.weight, 0.0)
                torch.nn.init.normal_(lin.weight[:, :self.in_dim], 0.0, np.sqrt(2) / np.sqrt(lin.out_features))
            elif self.encoding is not None and j in self.skip_connections:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(lin.out_features))
                torch.nn.init.constant_(lin.weight[:, self.in_dim:self.mlp_in_dim], 0.0)
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(lin.out_features))
            self.layers[j] = nn.utils.parametrizations.weight_norm(lin)
            

    def forward(self, in_tensor: Tensor) -> Tensor:
        in_tensor = self.encoding(in_tensor) if self.encoding is not None else in_tensor
        x = in_tensor
        for i, layer in enumerate(self.layers):
            if i in self._skip_connections:
                x = torch.cat([in_tensor, x], dim=-1)
            x = layer(x)
            if self.activation is not None and i < len(self.layers) - 1:
                x = self.activation(x)
        if self.out_activation:
            x = self.activation(x)
        return x




class SDF(nn.Module):

    """
    Abstract base classes for torch-based signed distance functions.
    """

    def __init__(self, dim: int):
        super(SDF, self).__init__()
        self.dim = dim

    def distance(self, x_in: Float[Tensor, "*batch in_dim"]) -> Float[Tensor, "*batch 1"]:
        raise NotImplementedError()
    
    def gradient(self, 
                 x_in: Float[Tensor, "*batch in_dim"],
                 dist: Optional[Float[Tensor, "*batch 1"]] = None,
                 differentiable: bool = False) -> Float[Tensor, "*batch 3"]:
        if dist is None:
            x = x_in.requires_grad_()
            dist = self.distance(x)

        d_outputs = torch.ones_like(dist)
        return ad.grad(dist, x_in, d_outputs, 
                       create_graph=differentiable, 
                       retain_graph=differentiable, 
                       only_inputs=True)[0]
    
    def project_nearest(self, 
                        x_in: Float[Tensor, "*batch in_dim"],
                        dist: Optional[Float[Tensor, "*batch 1"]] = None,
                        grad: Optional[Float[Tensor, "*batch 3"]] = None,
                        differentiable: bool = False) -> Float[Tensor, "*batch 3"]:
        x = x_in.requires_grad_() if grad is None else x_in
        if dist is None:
            dist = self.distance(x)
        if grad is None:
            grad = self.gradient(x, dist, differentiable)
        return x_in - dist * F.normalize(grad, dim=-1)
    
    def project_level_sets(self, 
                           x_in: Float[Tensor, "*batch sample 3"],
                           origin_dist: Optional[Float[Tensor, "*batch 1"]]
                           ) -> Float[Tensor, "*batch 3"]:
        x = x_in.requires_grad_()
        dist = self.distance(x)
        grad = self.gradient(x, dist)
        return x_in - (dist - origin_dist.unsqueeze(1)) * F.normalize(grad, dim=-1)
    
    def sphere_trace(self, 
                     x_in: Float[Tensor, "*batch 3"],
                     dir: Float[Tensor, "*batch 3"]) -> Float[Tensor, "*batch 3"]:
        dist = self.distance(x_in)
        return x_in - dist * dir
    
    def tangent_plane(self, 
                      x_in: Float[Tensor, "*batch in_dim"],
                      grad: Optional[Float[Tensor, "*batch 3"]] = None,
                      differentiable: bool = False) -> Float[Tensor, "*batch 3 3"]:
        if grad is None:
            x = x_in.requires_grad_() if grad is None else x_in
            dist = self.distance(x)
            grad = self.gradient(x, dist, differentiable)
        d = x_in.shape[-1]
        normal = F.normalize(grad, dim=-1)
        I = torch.eye(d, device=x_in.device).view(*([1] * (x_in.dim() - 1)), d, d).expand(*x_in.shape[:-1], -1, -1)
        z = I[..., 2]
        v = torch.linalg.cross(normal, z, dim=-1)
        c = (z * normal).sum(dim=-1)
        cross_matrix = cross_skew_matrix(v)
        scale = ((1 - c) / v.norm(dim=-1).pow(2)).view(-1, 1, 1)
        return I + cross_matrix + (cross_matrix @ cross_matrix) * scale
        # return F.normalize(I - (normal.unsqueeze(-2) * normal.unsqueeze(-1)), dim=-2)

    def sample_zero_level_set(self,
                              num_samples: int,
                              threshold: float = 0.05,
                              samples_per_step: int = 1000000,
                              bounds: Tuple[float, float] = (-1, 1),
                              num_projections: int = 1):
        n_samples = 0
        sampled_pts = []
        device = next(self.parameters()).device
        iterations = 0
        with torch.no_grad():
            while n_samples < num_samples:
                iterations += 1
                unif = torch.rand(samples_per_step, 3, device=device) * (bounds[1] - bounds[0]) + bounds[0]
                sdf = self.distance(unif)
                close = unif[sdf.squeeze() < threshold, :]
                sampled_pts.append(close)
                n_samples += close.shape[0]
        sampled_pts = torch.cat(sampled_pts, dim=0)[:num_samples, :]
        print(f"Sampled {sampled_pts.shape[0]} points in {iterations} iterations.")
        for it in range(num_projections):
            sampled_pts = self.project_nearest(sampled_pts)
        return sampled_pts
    
    
    
@dataclass
class NeuralSDFConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: NeuralSDF)
    network: MLPConfig = field(default_factory=MLPConfig)
    # network: MLPConfig = MLPConfig(
    #     in_dim=3,
    #     num_layers=8,
    #     layer_width=256,
    #     out_dim=1,
    #     skip_connections=(4,),
    #     activation='Softplus',
    #     act_defaults={'beta': 100},
    #     num_frequencies=6,
    #     encoding_with_input=True,
    #     geometric_init=True
    # )

class NeuralSDF(SDF):

    def __init__(self, config: NeuralSDFConfig, network: nn.Module):
        super(NeuralSDF, self).__init__(config.network.in_dim)
        self.config = config
        self.network = network

    def distance(self, x_in: Float[Tensor, "*batch in_dim"]) -> Float[Tensor, "*batch 1"]:
        return self.network(x_in)

    def forward(self, 
                x_in: Float[Tensor, "*batch in_dim"],
                with_grad: bool = False,
                differentiable_grad: bool = False) -> Dict[str, Float[Tensor, "*batch f"]]:
        outputs = {}
        x = x_in
        if with_grad:
            x = x.requires_grad_()
        dist = self.distance(x)
        outputs['dist'] = dist
        if with_grad:
            grad = self.gradient(x, dist, differentiable_grad)
            outputs['grad'] = grad
        return outputs


@dataclass
class MeshDataConfig:
    filename: Optional[str] = None
    path: Optional[Path] = None
    surf_sample: int = 16384
    space_sample: int = 16384
    uniform_ratio: float = 0.125
    local_noise_scale: float = 0.05
    device: str = 'cuda:1'


    @property
    def file(self) -> Path:
        if self.path is not None:
            return self.path
        elif self.filename is not None:
            return Path('data') / self.filename
        else:
            raise ValueError("Either filename or path must be provided.")

class MeshData:
    def __init__(self, config: MeshDataConfig):
        self.mesh = trimesh.load(config.file, force='mesh')
        self.verts = torch.tensor(self.mesh.vertices, dtype=torch.float32)
        self.faces = torch.tensor(self.mesh.faces, dtype=torch.long)

        # Center and normalize the mesh
        centroid = torch.tensor(self.mesh.centroid, dtype=torch.float32)
        self.verts -= centroid
        self.verts /= self.verts.abs().max()
        self.verts *= 0.8

        # Store geometry as batched Kaolin mesh
        self.geometry = kal.rep.SurfaceMesh(self.verts, self.faces).to_batched().to(config.device)

        self.device = config.device
        self.surf_sample = config.surf_sample
        self.space_sample = config.space_sample
        self.uniform_ratio = config.uniform_ratio
        self.noise_scale = config.local_noise_scale

        self.domain = torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32).to(config.device)
        self.dataset = MeshDataset(self)



class MeshDataset(Dataset):
    def __init__(self, mesh: MeshData):
        self.mesh = mesh

    def __len__(self):
        return 1  # Single mesh, stateless resampling per iteration

    def __getitem__(self, idx) -> Dict[str, Float[Tensor, "*batch 3"]]:
        # Surface samples
        surface_pts, face_ids = kal.ops.mesh.sample_points(
            self.mesh.geometry.vertices, self.mesh.geometry.faces, self.mesh.surf_sample)
        surface_pts = surface_pts.squeeze(0)
        face_ids = face_ids.squeeze(0)

        normals = self.mesh.geometry.face_normals.mean(dim=-2)[0, face_ids]

        # Space samples: Gaussian near-surface + uniform
        n_uniform = int(self.mesh.space_sample * self.mesh.uniform_ratio)
        n_gauss = self.mesh.space_sample - n_uniform

        unif = torch.rand(n_uniform, 3, device=self.mesh.device)
        unif = unif * (self.mesh.domain[1] - self.mesh.domain[0]) + self.mesh.domain[0]

        gauss = surface_pts[:n_gauss] + torch.randn(n_gauss, 3, device=self.mesh.device) * self.mesh.noise_scale

        space_pts = torch.cat([gauss, unif], dim=0)

        return {
            'surface_sample': surface_pts,
            'normal_sample': normals,
            'space_sample': space_pts
        }



@dataclass
class IGRConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: ImplicitGeometricRegularization)
    zero_sdf_surface_w: float = 1.0
    eikonal_error_w: float = 0.01
    normals_error_w: float = 0.01
    zero_penalty_w: float = 0.05


class ImplicitGeometricRegularization(nn.Module):

    def __init__(self, config: IGRConfig):
        super(ImplicitGeometricRegularization, self).__init__()
        self.config = config
        self.zero_loss = ZeroLoss()
        self.zero_penalty = ZeroPenalty()
        self.eikonal_loss = EikonalLoss()
        self.normals_loss = NormalsLoss()

    def forward(self, 
                pred_sdf_surf: Float[Tensor, "*batch 1"],
                pred_sdf_space: Float[Tensor, "*batch 1"],
                grad_sdf_surf: Float[Tensor, "*batch 3"],
                grad_sdf_space: Float[Tensor, "*batch 3"],
                surf_normals: Float[Tensor, "*batch 3"]
                ) -> Dict[str, Float[Tensor, "1"]]:
        zero_loss = self.config.zero_sdf_surface_w * self.zero_loss(pred_sdf_surf)
        zero_penalty = self.config.zero_penalty_w * self.zero_penalty(pred_sdf_space)
        eik_loss = self.config.eikonal_error_w * self.eikonal_loss(torch.cat([grad_sdf_space, grad_sdf_surf], dim=1))
        normals_loss = self.config.normals_error_w * self.normals_loss(grad_sdf_surf, surf_normals)
        return {'zero_loss': zero_loss, 
                'eikonal_loss': eik_loss, 
                'zero_penalty': zero_penalty,
                'normals_loss': normals_loss}


class EikonalLoss(nn.Module):

    def __init__(self):
        super(EikonalLoss, self).__init__()

    def forward(self, grad: Float[Tensor, "*batch 3"]) -> Float[Tensor, "1"]:
        return (grad.norm(dim=-1) - 1.0).abs().mean()
    

class NormalsLoss(nn.Module):

    def __init__(self):
        super(NormalsLoss, self).__init__()

    def forward(self, 
                grad: Float[Tensor, "*batch 3"],
                normals: Float[Tensor, "*batch 3"]) -> Float[Tensor, "1"]:
        return (1 - F.cosine_similarity(grad, normals, dim=-1)).mean()
    
    
class ZeroLoss(nn.Module):

    def __init__(self):
        super(ZeroLoss, self).__init__()

    def forward(self, dist: Float[Tensor, "*batch 1"]) -> Float[Tensor, "1"]:
        return dist.abs().mean()
    

class ZeroPenalty(nn.Module):

    def __init__(self, scale: float=100):
        super(ZeroPenalty, self).__init__()
        self.scale = scale

    def forward(self, dist: Float[Tensor, "*batch 1"]) -> Float[Tensor, "1"]:
        error = torch.exp(-self.scale * dist.abs())
        return error.mean()
    


@dataclass
class DeformationLossConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: DeformationLoss)
    arap_loss_w: float = 1.0
    moving_handle_loss_w: float = 1.0
    static_handle_loss_w: float = 1.0


class DeformationLoss(nn.Module):

    def __init__(self, config: DeformationLossConfig):
        super(DeformationLoss, self).__init__()
        self.config = config
        self.handle_loss = nn.MSELoss()

    def forward(self,
                patch_verts: Float[Tensor, "p n 3"],
                faces: Int[Tensor, "m 3"],
                rotations: Float[Tensor, "p n 3 3"],
                translations: Float[Tensor, "p n 3"],
                moving_idx: Int[Tensor, "h_1 2"],
                static_idx: Int[Tensor, "h_2 2"],
                moving_gt: Float[Tensor, "h_1 3"],
                static_gt: Float[Tensor, "h_2 3"]
                ) -> Float[Tensor, "1"]:
        patch_arap_loss = self.arap_loss(patch_verts, faces, rotations, translations)
        transformed_verts = (rotations @ patch_verts[..., None]).squeeze(-1) + translations
        moving_pos = transformed_verts[moving_idx[:, 0], moving_idx[:, 1], :]
        static_pos = transformed_verts[static_idx[:, 0], static_idx[:, 1], :]
        moving_handle_loss = self.handle_loss(moving_pos, moving_gt) if moving_gt.shape[0] > 0 else 0.0
        static_handle_loss = self.handle_loss(static_pos, static_gt) if static_gt.shape[0] > 0 else 0.0

        return {
            'arap_loss': patch_arap_loss * self.config.arap_loss_w,
            'moving_handle_loss': moving_handle_loss * self.config.moving_handle_loss_w,
            'static_handle_loss': static_handle_loss * self.config.static_handle_loss_w,
        }


def compute_gradient(inputs: Tensor, outputs: Tensor) -> Tensor:
    grads = grad(outputs=outputs,
                 inputs=inputs,
                 grad_outputs=torch.ones_like(outputs),
                 create_graph=True,
                 retain_graph=True,
                 only_inputs=True)[0]
    return grads


def detach_model(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False
        

def make_volume(device):
    steps = torch.linspace(
        min_coord,
        max_coord, 
        resolution,
        device=device
    )
    xx, yy, zz = torch.meshgrid(steps, steps, steps, indexing="ij")
    return torch.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T.float()


def world_to_grid(coord, min_coord, max_coord, resolution):
    """
    Maps a world-space point to voxel index space.
    """
    normalized = (coord - min_coord) / (max_coord - min_coord)
    index = normalized * (resolution - 1)
    return np.round(index).astype(int)


@torch.no_grad()
def evaluate_model(model, pts_volume):    
    f_eval = []
    for sample in tqdm(torch.split(pts_volume, chunk, dim=0)):
        sdf_vals = model(sample.contiguous())[:, 0]
        f_eval.append(sdf_vals.cpu())
    f_volume = torch.cat(f_eval, dim=0).reshape(resolution, resolution, resolution)
    return f_volume


def extract_mesh(sdf, resolution, level):
    try:
        verts, faces = mcubes.marching_cubes(sdf.numpy(), level)
        verts = verts / (resolution - 1) * 2 - 1
    except Exception as e:
        print(f"Marching cubes failed: {e}")
        verts = np.empty([0, 3], dtype=np.float32)
        faces = np.empty([0, 3], dtype=np.int32)
    
    return verts, faces


def plot_mesh(verts, faces, save_path=None):
    x, y, z = verts.T
    i, j, k = faces.T

    mesh_plot = go.Figure(data=[
        go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color='lightblue',
            opacity=0.5,
            flatshading=True,
        )
    ])

    mesh_plot.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
        ),
        title="SDF Visualization",
        width=800,
        height=800
    )

    if save_path is not None:
        mesh_plot.write_html(f"{save_path}.html")
        mesh_plot.write_image(f"{save_path}.png")
    else:
        mesh_plot.show()


def train(config: MeshDataConfig):
    device = config.device
    epochs = 10000

    mlp_cfg = MLPConfig()
    sdf_model = MLP(mlp_cfg).to(device)

    igr_cfg = IGRConfig()
    igr_loss = ImplicitGeometricRegularization(igr_cfg)

    optimizer = torch.optim.Adam(sdf_model.parameters(), lr=1e-4)

    mesh_data = MeshData(config)
    dataset = mesh_data.dataset
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for epoch in trange(epochs, desc="Epochs"):
        for batch in dataloader:
            surface_pts = batch['surface_sample'].squeeze(0).to(device).requires_grad_()
            surface_normals = batch['normal_sample'].squeeze(0).to(device)
            space_pts = batch['space_sample'].squeeze(0).to(device).requires_grad_()

            pred_sdf_surface = sdf_model(surface_pts).squeeze(-1).unsqueeze(-1)
            pred_sdf_space = sdf_model(space_pts).squeeze(-1).unsqueeze(-1)

            grad_sdf_surface = compute_gradient(surface_pts, pred_sdf_surface)
            grad_sdf_space = compute_gradient(space_pts, pred_sdf_space)

            loss_dict = igr_loss(pred_sdf_surface, pred_sdf_space,
                                grad_sdf_surface, grad_sdf_space,
                                surface_normals)

            total_loss = sum(loss_dict.values())

            if epoch % 100 == 0:
                tqdm.write(f"Epoch {epoch}, Losses: " + ", ".join(f"{k}: {v.item():.6f}" for k, v in loss_dict.items()))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        os.makedirs('out/SDFs', exist_ok=True)
        os.makedirs(f'out/SDFs/{config.file.stem}', exist_ok=True)
        sdf_path = f'out/SDFs/{config.file.stem}/{config.file.stem}-SDF.pth'
        torch.save(sdf_model.state_dict(), sdf_path)


def eval(config: MeshDataConfig):
    device = config.device
    mlp_cfg = MLPConfig()
    sdf_model = MLP(mlp_cfg).to(device)

    sdf_path = f'out/SDFs/{config.file.stem}/{config.file.stem}-SDF.pth'
    sdf_model.load_state_dict(torch.load(sdf_path))
    sdf = evaluate_model(sdf_model, make_volume(device))

    mesh_data = MeshData(config)

    landmarks_voxels = []
    if landmarks_indices is not None and len(landmarks_indices) > 0:
        landmarks_vertices = mesh_data.verts[landmarks_indices]
        for landmark_vertex in landmarks_vertices:
            landmark_vertex = landmark_vertex.cpu().numpy()
            voxel_index = world_to_grid(landmark_vertex, min_coord, max_coord, resolution)
            print(f"> Landmark vertex {landmark_vertex} mapped to voxel index {voxel_index}")
            landmarks_voxels.append(voxel_index)
        landmarks_path = f'out/SDFs/{config.file.stem}/{config.file.stem}'
        np.save(f'{landmarks_path}-landmarks-voxels.npy', landmarks_voxels)
        np.save(f'{landmarks_path}-landmarks-vertices.npy', landmarks_vertices.cpu().numpy())
        print(f"Landmarks voxel indices saved to {landmarks_path}-landmarks-voxels.npy")
        print(f"Landmarks vertices saved to {landmarks_path}-landmarks-vertices.npy")

    verts, faces = extract_mesh(sdf, resolution=resolution, level=0.0)
    plot_mesh(verts, faces, save_path=f'out/SDFs/{config.file.stem}/SDF-{config.file.stem}')
    print(f"Extracted mesh with {len(verts)} vertices and {len(faces)} faces.")
    print(f"Mesh plot saved to out/SDFs/{config.file.stem}/SDF-{config.file.stem}.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate SDF model on mesh")
    parser.add_argument('--filename', type=str, help="Filename in ./data directory")
    parser.add_argument('--path', type=str, help="Full path to mesh file")
    args = parser.parse_args()

    mesh_config = MeshDataConfig(
        filename=args.filename if args.path is None else None,
        path=Path(args.path) if args.path is not None else None
    )

    print(f"Using mesh file: {mesh_config.file}")
    print("Starting SDF training...")
    train(mesh_config)
    print("Training completed.")
    print("Starting SDF evaluation...")
    eval(mesh_config)
    print("Done.")
