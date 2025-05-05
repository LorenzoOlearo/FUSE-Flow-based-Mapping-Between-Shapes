"""
In this file we define the model used for training the flow matching model and the Geometry distribution model.
"""

import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional
from networks import *

from flow_matching.solver import ODESolver



######################   GEOMDIST MODEL   ##########################

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class EDMPrecond(torch.nn.Module):
    def __init__(self,
        channels = 3, 
        use_fp16 = False,
        sigma_min = 0,
        sigma_max = float('inf'),
        sigma_data  = 1,
        depth = 6,
        network = None,
    ):
        super().__init__()

        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.sigma_data = sigma_data
        if network is not None:
            self.model = network
        else:
            self.model = MLP(channels=channels, hidden_size=512, depth=depth)

    def forward(self, x, sigma, force_fp32=False, **model_kwargs):

        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
    
        F_x = self.model((c_in * x).to(dtype), c_noise, **model_kwargs).to(dtype)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)

        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    #@torch.no_grad()
    def sample(self, cond=None, batch_seeds=None, channels=3, num_steps=18):

        device = batch_seeds.device
        batch_size = batch_seeds.shape[0]

        rnd = None
        points = batch_seeds

        latents = points.float().to(device)

        points = edm_sampler(self, latents, cond, num_steps=num_steps)
        return points

    #@torch.no_grad()
    def inverse(self, cond=None, samples=None, channels=3, num_steps=18):
        return inverse_edm_sampler(self, samples, cond, num_steps=num_steps)


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):  
    # disable S_churn
    assert S_churn==0

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    outputs = []
    outputs.append((x_next / t_steps[0]).detach().cpu().numpy())
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        # x_hat = x_cur
        t_hat = t_cur

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        outputs.append((x_next / (1+t_next**2).sqrt()).detach().cpu().numpy())
    return x_next, outputs

def inverse_edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):  
    # disable S_churn
    assert S_churn==0

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])+1e-8]) # t_N = 0
    t_steps = torch.flip(t_steps, [0])#[1:]

    # Main sampling loop.
    x_next = latents.to(torch.float64)# * t_steps[0]

    # outputs = []
    outputs = None
    # outputs.append((x_next / t_steps[0]).detach().cpu().numpy())

    #print(t_steps[0])
    #print(x_next.mean(), x_next.std())
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        # print('steps', t_cur, t_next)
        x_cur = x_next
        # print('cur', (x_cur / t_cur).mean(), (x_cur / t_cur).std())

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        x_hat = x_cur
        t_hat = t_cur

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        #print('next', (x_next / (1+t_next**2).sqrt()).mean(), (x_next / (1+t_next**2).sqrt()).std())

        # outputs.append((x_next / (1+t_next**2).sqrt()).detach().cpu().numpy())
    x_next = x_next / (1+t_next**2).sqrt()
    return x_next, outputs


######################   FLOW MATCHING MODEL   ##########################

class FMCond(torch.nn.Module):
    def __init__(self,
        channels = 3, 
        use_fp16 = False,
        sigma_min = 0,
        sigma_max = float('inf'),
        sigma_data  = 1,
        depth = 6,
        network = None,
    ):
        super().__init__()

        self.use_fp16 = use_fp16

        if network is not None:
            self.net = network
        else:
            self.net = MLP(channels=channels, hidden_size=512, depth=depth)

    def forward(self, x, sigma):
        x = x
        sigma = sigma.to(torch.float32).reshape(-1, 1)
        V_x = self.net(x, sigma).to(torch.float32)

        return V_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    #@torch.no_grad()
    def sample(self, noise=None, num_steps=64,enable_grad=False, intermediate=False):

        device = noise.device

        noise = noise.float().to(device)

        if intermediate:
            sample,sol=ot_sampler(self.net, noise, num_steps=num_steps,enable_grad=enable_grad, intermediate=intermediate)
            return sample, sol
        else:
            sample=ot_sampler(self.net, noise, num_steps=num_steps,enable_grad=enable_grad,intermediate=intermediate)
            return sample
    
    def inverse(self, samples=None, num_steps=64,enable_grad=False,intermediate=False):

        device = samples.device
        samples = samples.float().to(device)

        if intermediate:
            sample,sol=ot_inverse(self.net, samples, num_steps=num_steps,enable_grad=enable_grad, intermediate=intermediate)
            
            return sample, sol
        else:
            sample=ot_inverse(self.net, samples, num_steps=num_steps,enable_grad=enable_grad, intermediate=intermediate)
            
            return sample



def ot_sampler(
    net, latents,num_steps=18, enable_grad=False, intermediate=False):  
    """Function to integrate a model from 0 to 1"""

    # Time step discretization.
    t_steps = torch.linspace(0, 1, num_steps+1)
    # Main sampling loop.
    solver = ODESolver(velocity_model=net)
    solutions = solver.sample(time_grid=t_steps, x_init=latents, method='midpoint', step_size=1/num_steps, return_intermediates=intermediate,enable_grad=enable_grad)

    if intermediate:
        return solutions[-1], solutions
    else:
        return solutions


class InverseModel(torch.nn.Module):
    """
    Inverse model for a 3d+t model."""
    def __init__(self, vector_field):
        super(InverseModel, self).__init__()
        self.vector_field = vector_field

    def forward(self, x,t):
        if torch.allclose(t, torch.ones_like(t)):
            return torch.zeros_like(x)
        return -self.vector_field(x,1-t)

def ot_inverse(    net, sample,
    num_steps=18, enable_grad=False, intermediate=False):
    """Function to integrate a model from 1 to 0"""
    
    # Time step discretization.
    t_steps = torch.linspace(0, 1, num_steps+1)
    # Main sampling loop.
    inverse_net=InverseModel(net)
    solver = ODESolver(velocity_model=inverse_net)
    solutions = solver.sample(time_grid=t_steps, x_init=sample, method='midpoint', step_size=1/num_steps, return_intermediates=intermediate,enable_grad=enable_grad)

    if intermediate:
        
        return solutions[-1], solutions
    else:
        return solutions
