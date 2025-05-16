"""
In this file we define some networks that takes as input a spatian input and a time input.
"""

import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F




###################### GEOMDIST NETWORKS ######################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)
    


def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)

def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596

def mp_sum(a, b, t=0.5):
    # print(a.mean(), a.std(), b.mean(), b.std())
    return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t ** 2)

class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w)) # forced weight normalization
        w = normalize(w) # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel())) # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1]//2,))

class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128, other_dim=0):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        # self.mlp = nn.Linear(self.embedding_dim+3, dim)/
        self.mlp = MPConv(self.embedding_dim+3+other_dim, dim, kernel=[])

    @staticmethod
    def embed(input, basis):
        # print(input.shape, basis.shape)
        projections = torch.einsum('nd,de->ne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=1)
        return embeddings
    
    def forward(self, input):
        # input: N x 3
        if input.shape[1] != 3:
            input, others = input[:, :3], input[:, 3:]
        else:
            others = None
        
        if others is None:
            embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=1)) # N x C
        else:
            embed = self.mlp(torch.cat([self.embed(input, self.basis), input, others], dim=1))
        return embed


class Network(nn.Module):
    def __init__(
        self,
        channels = 3,
        hidden_size = 512,
        depth = 6,
    ):
        super().__init__()

        self.emb_fourier = MPFourier(hidden_size)
        self.emb_noise = MPConv(hidden_size, hidden_size, kernel=[])

        self.x_embedder = PointEmbed(dim=hidden_size, other_dim=channels-3)

        self.gains = nn.ParameterList([
            torch.nn.Parameter(torch.zeros([])) for _ in range(depth)
        ])
        ##
        self.layers = nn.ModuleList([
            nn.ModuleList([
                MPConv(hidden_size, hidden_size, []),
                MPConv(hidden_size, hidden_size, []),
                MPConv(hidden_size, 1 * hidden_size, []),
            ]) for _ in range(depth)
        ])


        self.final_emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.final_out_gain = torch.nn.Parameter(torch.zeros([]))
        self.final_layer = nn.ModuleList([
            MPConv(hidden_size, hidden_size, []),
            MPConv(hidden_size, channels, []),
            MPConv(hidden_size, hidden_size, []),
        ])

        self.res_balance = 0.3


    def forward(self, x, t):
        x = self.x_embedder(x)
        if t.ndim == 1:
            t = t.repeat(x.shape[0])

        t = mp_silu(self.emb_noise(self.emb_fourier(t.flatten())))

        for (x_proj_pre, x_proj_post, emb_linear), emb_gain in zip(self.layers, self.gains):

            c = emb_linear(t, gain=emb_gain) + 1

            x = normalize(x)
            y = x_proj_pre(mp_silu(x))
            y = mp_silu(y * c.to(y.dtype))
            y = x_proj_post(y)
            x = mp_sum(x, y, t=self.res_balance)

        x_proj_pre, x_proj_post, emb_linear = self.final_layer
        c = emb_linear(t, gain=self.final_emb_gain) + 1
        y = x_proj_pre(mp_silu(normalize(x)))
        y = mp_silu(y * c.to(y.dtype))
        out = x_proj_post(y, gain=self.final_out_gain)
    
        return out
    
    


#######################   MLP  NETWORKS  #######################
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x): 
        return torch.sigmoid(x) * x


class RandomFourierFeatures(nn.Module):
    def __init__(self, input_dim, output_dim, scale=1.0):
        super(RandomFourierFeatures, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scale = scale
        self.B = nn.Parameter(self.scale * torch.randn(self.input_dim, self.output_dim // 2), requires_grad=False)
        

    def forward(self, x):
        x_proj = x @ self.B
        x_proj = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return x_proj


class FourierFeatsEncoding(nn.Module):
    """Multi-scale sinusoidal encodings. Support ``integrated positional encodings`` if covariances are provided.
    Each axis is encoded with frequencies ranging from 2^min_freq_exp to 2^max_freq_exp.

    Args:
        in_dim: Input dimension of tensor
        num_frequencies: Number of encoded frequencies per axis
        include_input: Append the input coordinate to the encoding
    """

    def __init__(
        self,
        in_dim: int,
        num_frequencies: int,
        include_input: bool = False
    ) -> None:
        super(FourierFeatsEncoding, self).__init__()

        assert in_dim > 0, "in_dim should be greater than zero"
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.min_freq = 0.0
        self.max_freq = num_frequencies - 1.0
        self.include_input = include_input

    def get_out_dim(self) -> int:
        assert self.in_dim is not None, "Input dimension has not been set"
        out_dim = self.in_dim * self.num_frequencies * 2
        if self.include_input:
            out_dim += self.in_dim
        return out_dim

    def forward(
        self,
        in_tensor
    ):
        """Calculates NeRF encoding. 

        Args:
            in_tensor: For best performance, the input tensor should be between 0 and 1.
        Returns:
            Output values will be between -1 and 1
        """
        scaled_in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
        freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies).to(in_tensor.device)
        scaled_inputs = scaled_in_tensor[..., None] * freqs  # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)  # [..., "input_dim" * "num_scales"]

        encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1))

        if self.include_input:
            encoded_inputs = torch.cat([in_tensor, encoded_inputs], dim=-1)
        return encoded_inputs


class MLP(nn.Module):
    """MLP 3D+time+features with Swish activations."""

    def __init__(self,
        channels = 3,
        hidden_size = 256,
        depth = 6,):
        #input_dim: int = 3, time_dim: int = 1, hidden_dim: int = 128, fourier_encoding: str = 'FF', fourier_dim: int = 0):
        super().__init__()
        self.input_dim = channels
        self.hidden_dim = hidden_size
        self.ff_module = FourierFeatsEncoding(in_dim=4, num_frequencies=6, include_input=True)
        self.fourier_dim = ((3+1) * 6 * 2) + (3 + 1) + (self.input_dim-3)
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
    

    def forward(self, x, t):
        sz = x.size()
        t = t.reshape(-1, 1)        
        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        xyz=x[:,:3]
        h = torch.cat([xyz, t], dim=1)
        
        h = self.rff_module(h)
        h = self.ff_module(h)
        output = self.main(torch.cat([h, x[:,3:]], dim=-1))
        
        return output

class MLP_general(nn.Module):
    """MLP 3D+time+features with Swish activations."""

    def __init__(self,
        channels = 3,
        hidden_size = 256,
        depth = 6,):
        #input_dim: int = 3, time_dim: int = 1, hidden_dim: int = 128, fourier_encoding: str = 'FF', fourier_dim: int = 0):
        super().__init__()
        self.input_dim = channels
        self.hidden_dim = hidden_size
        self.ff_module = FourierFeatsEncoding(in_dim=4, num_frequencies=6, include_input=True)
        self.fourier_dim = ((channels+1) * 6 * 2) + (channels + 1) 
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
    

    def forward(self, x, t):
        sz = x.size()
        t = t.reshape(-1, 1)        
        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        h = torch.cat([x, t], dim=1)
        
        h = self.rff_module(h)
        h = self.ff_module(h)
        output = self.main(h)
        
        return output




############################ TINY CUDA NETWORKS ############################


try:
    import tinycudann as tcnn
    print("tinycudann is available.")
except ImportError:
    print("tinycudann is not available.")

class MLP_tiny(nn.Module):
    """MLP 3D+time+features with Swish activations and tinycudann."""
    def __init__(self,
        channels = 3,
        hidden_size = 256,
        depth = 6,):
        super().__init__()
        self.input_dim = channels
        self.hidden_dim = hidden_size
        
        self.config={
            "encoding": {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 16,
                "base_resolution": 16,
                "per_level_scale": 2
            },
            "network": {
                "otype": "FullyFusedMLP",
                "activation": "ReLu",
                "output_activation": "None",
                "n_neurons": 128,
                "n_hidden_layers": 6
            }
        }
        self.fourier_dim = ((3+1) * 6 * 2) + (3 + 1)
        self.ff_module = FourierFeatsEncoding(in_dim=4, num_frequencies=6, include_input=True)
        self.rff_module = nn.Identity()


        self.encoding = tcnn.Encoding(4, self.config["encoding"])

    
        self.main = nn.Sequential(
                nn.Linear(self.encoding.n_output_dims+ self.fourier_dim, self.hidden_dim),
                Swish(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                Swish(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                Swish(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                Swish(),
                nn.Linear(self.hidden_dim, self.input_dim),
            )
    def forward(self, x, t):
        sz = x.size()
        t = t.reshape(-1, 1)        
        t = t.reshape(-1, 1).expand(x.shape[0], 1)

        h = torch.cat([x, t], dim=1)
        
        h_ff = self.ff_module(h)
        h_ngp=self.encoding(h).to(torch.float32)
        
        output = self.main(torch.cat([h_ff, h_ngp], dim=1))
        output = output.reshape(*sz)
        
        return output


