"""
In this file we define the model used for training the flow matching model and the Geometry distribution model.
"""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
from flow_matching.solver import ODESolver
from torch import Tensor
from tqdm import tqdm

from .networks import *

######################   GEOMDIST MODEL   ##########################


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class EDMPrecond(torch.nn.Module):
    def __init__(
        self,
        channels=3,
        use_fp16=False,
        sigma_min=0,
        sigma_max=float("inf"),
        sigma_data=1,
        depth=6,
        network=None,
    ):
        super().__init__()

        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.sigma_data = sigma_data
        if network is not None:
            self.net = network
        else:
            self.net = MLP(channels=channels, hidden_size=512, depth=depth)

    def forward(self, x, sigma, force_fp32=False, **model_kwargs):

        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1)
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.net((c_in * x).to(dtype), c_noise, **model_kwargs).to(dtype)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)

        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    # @torch.no_grad()
    def sample(self, cond=None, noise=None, num_steps=18, intermediate=False):

        device = noise.device
        batch_size = noise.shape[0]

        rnd = None
        points = noise

        latents = points.float().to(device)

        points = edm_sampler(
            self, latents, cond, num_steps=num_steps, intermediate=intermediate
        )
        return points

    # @torch.no_grad()
    def inverse(self, cond=None, samples=None, num_steps=18, intermediate=False):
        return inverse_edm_sampler(
            self, samples, cond, num_steps=num_steps, intermediate=intermediate
        )


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds
        ]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators]
        )

    def randn_like(self, input):
        return self.randn(
            input.shape, dtype=input.dtype, layout=input.layout, device=input.device
        )

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [
                torch.randint(*args, size=size[1:], generator=gen, **kwargs)
                for gen in self.generators
            ]
        )


def edm_sampler(
    net,
    latents,
    class_labels=None,
    randn_like=torch.randn_like,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
    intermediate=False,
):
    # disable S_churn
    assert S_churn == 0

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    )  # t_N = 0
    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    outputs = []
    outputs.append((x_next / t_steps[0]).detach().cpu().numpy())
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        )
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)
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
        outputs.append((x_next / (1 + t_next**2).sqrt()).detach().cpu().numpy())
    if intermediate:
        return x_next, outputs
    else:
        return x_next


def inverse_edm_sampler(
    net,
    latents,
    class_labels=None,
    randn_like=torch.randn_like,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
    intermediate=False,
):
    # disable S_churn
    assert S_churn == 0

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1]) + 1e-8]
    )  # t_N = 0
    t_steps = torch.flip(t_steps, [0])  # [1:]

    # Main sampling loop.
    x_next = latents.to(torch.float64)  # * t_steps[0]

    # outputs = []
    outputs = None
    # outputs.append((x_next / t_steps[0]).detach().cpu().numpy())

    # print(t_steps[0])
    # print(x_next.mean(), x_next.std())
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        # print('steps', t_cur, t_next)
        x_cur = x_next
        # print('cur', (x_cur / t_cur).mean(), (x_cur / t_cur).std())

        # Increase noise temporarily.
        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        )
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)
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

        # print('next', (x_next / (1+t_next**2).sqrt()).mean(), (x_next / (1+t_next**2).sqrt()).std())

        # outputs.append((x_next / (1+t_next**2).sqrt()).detach().cpu().numpy())
    x_next = x_next / (1 + t_next**2).sqrt()
    if intermediate:
        return x_next, outputs
    else:
        return x_next


######################   FLOW MATCHING MODEL   ##########################


class FMCond(torch.nn.Module):
    def __init__(
        self,
        channels=3,
        use_fp16=False,
        depth=6,
        network=None,
        use_edm_preconditioning=False,
        sigma_data=1,
    ):
        super().__init__()

        self.use_fp16 = use_fp16
        self.use_edm_preconditioning = use_edm_preconditioning
        self.sigma_data = sigma_data

        if network is not None:
            self.net = network
        else:
            print(
                f"[WARNING] Using default MLP with channels={channels}, hidden_size=512, depth={depth}"
            )
            self.net = MLP(channels=channels, hidden_size=512, depth=depth)

    def forward(self, x, t):
        t = t.to(torch.float32).reshape(-1, 1)

        if self.use_edm_preconditioning:
            c_skip = self.sigma_data**2 / (t**2 + self.sigma_data**2)
            c_out = t * self.sigma_data / (t**2 + self.sigma_data**2).sqrt()
            c_in = 1 / (self.sigma_data**2 + t**2).sqrt()
            c_noise = t.clamp(min=1e-8).log() / 4
            F_x = self.net(c_in * x, c_noise).to(torch.float32)
            V_x = c_skip * x + c_out * F_x
        else:
            V_x = self.net(x, t).to(torch.float32)

        return V_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def sample(self, noise, num_steps, enable_grad=False, intermediate=False):
        device = noise.device
        noise = noise.float().to(device)

        if intermediate:
            sample, sol = ot_sampler(
                self,
                noise,
                num_steps=num_steps,
                enable_grad=enable_grad,
                intermediate=intermediate,
            )
            return sample, sol
        else:
            tqdm.write(f"[INTEGRATION] Integrating forward with num_steps: {num_steps}")
            sample = ot_sampler(
                self,
                noise,
                num_steps=num_steps,
                enable_grad=enable_grad,
                intermediate=intermediate,
            )
            return sample

    def inverse(self, samples, num_steps, enable_grad=False, intermediate=False):
        device = samples.device
        samples = samples.float().to(device)

        if intermediate:
            sample, sol = ot_inverse(
                self,
                samples,
                num_steps=num_steps,
                enable_grad=enable_grad,
                intermediate=intermediate,
            )
            return sample, sol

        else:
            tqdm.write(
                f"[INTEGRATION] Integrating backward with num_steps: {num_steps}"
            )
            sample = ot_inverse(
                self,
                samples,
                num_steps=num_steps,
                enable_grad=enable_grad,
                intermediate=intermediate,
            )
            return sample


def ot_sampler(net, latents, num_steps, enable_grad=False, intermediate=False):
    """Function to integrate a model from 0 to 1"""
    t_steps = torch.linspace(0, 1, num_steps + 1)
    solver = ODESolver(velocity_model=net)
    solutions = solver.sample(
        time_grid=t_steps,
        x_init=latents,
        method="midpoint",
        step_size=1 / num_steps,
        return_intermediates=intermediate,
        enable_grad=enable_grad,
    )

    if intermediate:
        return solutions[-1], solutions
    else:
        return solutions


# class FMCond(nn.Module):
#     def __init__(
#         self,
#         channels: int = 3,
#         use_fp16: bool = False,
#         depth: int = 6,
#         network: Optional[nn.Module] = None,
#     ) -> None:
#         super().__init__()

#         self.use_fp16 = use_fp16

#         # Use provided network or default MLP
#         self.net: nn.Module = (
#             network if network is not None else MLP(channels=channels, hidden_size=512, depth=depth)
#         )

#     def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
#         """
#         Forward pass for conditional velocity model.

#         Parameters
#         ----------
#         x : Tensor
#             Input latent tensor.
#         sigma : Tensor
#             Conditioning noise level.

#         Returns
#         -------
#         Tensor
#             Output velocity V(x, sigma).
#         """
#         sigma = sigma.to(torch.float32).reshape(-1, 1)
#         return self.net(x, sigma).to(torch.float32)

#     def round_sigma(self, sigma: Tensor) -> Tensor:
#         """Round/convert sigma to tensor."""
#         return torch.as_tensor(sigma)

#     def sample(
#         self,
#         noise: Tensor,
#         num_steps: int = 64,
#         enable_grad: bool = False,
#         return_intermediates: bool = False,
#         solver_method: str = "midpoint",
#     ) -> Tuple[Tensor, Optional[List[Tensor]]]:
#         """
#         Forward integration from noise to sample.
#         """

#         final, intermediates = integrate_model(
#             net=self.net,
#             latents=noise,
#             num_steps=num_steps,
#             enable_grad=enable_grad,
#             return_intermediates=return_intermediates,
#             solver_method=solver_method,
#         )

#         return final, intermediates

#     def inverse(
#         self,
#         samples: Tensor,
#         num_steps: int = 64,
#         enable_grad: bool = False,
#         return_intermediates: bool = False,
#         solver_method: str = "midpoint",
#     ) -> Tuple[Tensor, Optional[List[Tensor]]]:
#         """
#         Reverse integration: map samples back to noise space.
#         """

#         samples = samples.to(torch.float32).to(samples.device)

#         final, intermediates = integrate_model(
#             net=self.net,
#             latents=samples,
#             num_steps=num_steps,
#             enable_grad=enable_grad,
#             return_intermediates=return_intermediates,
#             solver_method=solver_method,
#         )

#         return final, intermediates


# def integrate_model(
#     net: torch.nn.Module,
#     latents: Tensor,
#     num_steps: int = 18,
#     enable_grad: bool = False,
#     return_intermediates: bool = False,
#     solver_method: str = "midpoint",
# ) -> Tuple[Tensor, Optional[List[Tensor]]]:
#     """
#     Integrate a velocity-based model from time t=0 to t=1.

#     Parameters
#     ----------
#     net : torch.nn.Module
#         The velocity model to integrate.
#     latents : torch.Tensor
#         Initial latent variables.
#     num_steps : int, optional
#         Number of discretization steps in [0, 1], by default 18.
#     enable_grad : bool, optional
#         Whether gradients should be enabled during integration.
#     return_intermediates : bool, optional
#         If True, return all intermediate solutions in addition to the final one.
#     solver_method : str, optional
#         Numerical ODE solver method name (e.g., "midpoint", "modeleuler", etc.).

#     Returns
#     -------
#     (torch.Tensor, Optional[List[torch.Tensor]])
#         A tuple containing:
#         - final solution (Tensor)
#         - list of intermediate solutions, or None if not requested
#     """

#     time_grid: Tensor = torch.linspace(0.0, 1.0, num_steps + 1).to(latents.device)

#     solver = ODESolver(velocity_model=net)
#     sol: List[Tensor] = solver.sample(
#         time_grid=time_grid,
#         x_init=latents,
#         method=solver_method,
#         step_size=1/num_steps,
#         return_intermediates=return_intermediates,
#         enable_grad=enable_grad,
#     )

#     if return_intermediates:
#         return sol[-1], sol
#     else:
#         return sol, None


class InverseModel(torch.nn.Module):
    """
    Inverse model for a 3d+t model."""

    def __init__(self, vector_field):
        super(InverseModel, self).__init__()
        self.vector_field = vector_field

    def forward(self, x, t):
        if torch.allclose(t, torch.ones_like(t)):
            return torch.zeros_like(x)
        return -self.vector_field(x, 1 - t)


def ot_inverse(net, sample, num_steps, enable_grad=False, intermediate=False):
    """Function to integrate a model from 1 to 0"""
    t_steps = torch.linspace(0, 1, num_steps + 1)
    inverse_net = InverseModel(net)
    solver = ODESolver(velocity_model=inverse_net)
    solutions = solver.sample(
        time_grid=t_steps,
        x_init=sample,
        method="midpoint",
        step_size=1 / num_steps,
        return_intermediates=intermediate,
        enable_grad=enable_grad,
    )

    if intermediate:

        return solutions[-1], solutions
    else:
        return solutions
