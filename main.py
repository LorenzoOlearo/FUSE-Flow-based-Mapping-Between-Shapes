"""
Main script for training and inference of the neural network model.
This script includes functions for training the model, performing inference, and setting up the data loader.
It also includes command-line argument parsing and logging setup.
"""

import argparse
import datetime
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from tqdm import trange

import numpy as np
import torch
import trimesh

from util import misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.train_utils import setup_logging, initialize_device_and_seed
from util.mesh_utils import (
    get_shape_diameter,
    generate_embeddings,
    compute_features,
    sample_initial_distribution,
)
import util.lr_sched as lr_sched

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath

from model import models, networks
from model.models import EDMPrecond, FMCond


def get_inline_arg():
    parser = argparse.ArgumentParser("Train", add_help=False)
    # common parameters
    parser.add_argument(
        "--config", default=None, type=str, help="Path to the config file"
    )  # Model parameters
    parser.add_argument(
        "--method", default="FM", type=str, help="Method used for training, either 'FM' for Flow Matching or 'diffusion'"
    )
    parser.add_argument(
        "--network", default="MLP", type=str, help="Network used for training"
    )
    parser.add_argument("--train", action="store_true", help="Train a <run_name> model")
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Perform inference on the <run_name> model",
    )
    parser.add_argument(
        "--device", default="cuda:1", help="device to use for training / testing"
    )
    parser.add_argument("--run_name", default="RUN", help="Name of the run")
    parser.add_argument("--embedding", default="xyz", type=str)
    parser.add_argument("--landmarks", default=[412, 5891, 6593, 3323, 2119], type=list)
    parser.add_argument("--seed", default=21, type=int)
    parser.add_argument("--epochs", default=10000, type=int)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--batch_size",
        default=10000,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )  # 1024*64*2

    parser.add_argument(
        "--num_points_inference",
        default=10000,
        type=int,
        help="Number of points for inference",
    )
    parser.add_argument(
        "--num_points_train",
        default=10000,
        type=int,
        help="Number of points for inference",
    )  # 2048 * 64 * 4 * 64

    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )
    parser.add_argument("--num-steps", default=64, type=int)
    parser.add_argument("--depth", default=6, type=int, metavar="MODEL")

    parser.add_argument(
        "--data_path",
        default="shapes/Jellyfish_lamp_part_A__B_normalized.obj",
        type=str,
        help="dataset path",
    )

    parser.add_argument(
        "--distribution",
        default="gaussian",
        choices=["gaussian", "uniform", "sphere"],
        type=str,
    )

    # Optimizer parameters
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    parser.add_argument(
        "--blr",
        type=float,
        default=5e-7,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=0.75,
        help="layer-wise lr decay from ELECTRA/BEiT",
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=5e-7,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )
    parser.add_argument(
        "--verbose", default=False, action="store_true", help="Verbose output"
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=0, metavar="N", help="epochs to warmup LR"
    )
    # Dataset parameters
    parser.add_argument("--intermediate", action="store_true")
    parser.add_argument(
        "--output_dir",
        default="./out/",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir", default="./output/", help="path where to tensorboard log"
    )
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor",
    )
    parser.add_argument("--num_workers", default=32, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=2, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=2, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    parser.add_argument(
        "--embedding_dim",
        default=3,
        type=int,
        nargs="+",
        help="Dimension of the features",
    )

    parser.add_argument(
        "--embedding_type",
        default=None,
        type=str,
        nargs="+",
        help="Type of embedding on which the model is trained (e.g., features_only, xyz, features_and_xyz)",
    )

    parser.add_argument(
        "--features_type",
        default="xyz",
        type=str,
        help="The type of features to use for each point"
    )

    parser.add_argument(
            "--features_path", 
            default=None,
            type=str,
            help="Path to the features file"
    )

    parser.add_argument(
            "--vertex_features_path", 
            default=None,
            type=str,
            help="Path to the vertex features file"
    )

    parser.add_argument(
        "--features_normalization",
        default="none",
        type=str,
        help="Normalization to apply to the features: none, 0_1_indipendent, 0_1_global, 0_center_indipendent, 0_center_global",
    )

    parser.add_argument(
        "--features_interpolation",
        type=int,
        default=-1,
        help="Number of points to sample for interpolating the features. If -1, no interpolation is performed and the features are used as they are.",
    )

    args = parser.parse_args()

    return args


def setup_data_loader(args):
    """Set up the data loader based on the input data path."""
    data_loader_train = {
        "obj_file": args.data_path,
        "batch_size": args.batch_size,
        "epoch_size": args.num_points_train // args.batch_size,
    }
    print(f"Data path: {args.data_path}")
    return data_loader_train


def normalize_features(features, vertex_features, method, diameter=None):
    """Normalize features based on the specified method."""
    if method == "none":
        print("No normalization applied to features.")
        normalized = features
        vertex_features_normalized = vertex_features

    elif method == "0_1_indipendent":
        # Normalize each feature channel to [0, 1]
        min_vals = vertex_features.min(dim=0).values
        max_vals = vertex_features.max(dim=0).values
        normalized = (features - min_vals) / (max_vals - min_vals + 1e-8)
        vertex_features_normalized = (vertex_features - min_vals) / (max_vals - min_vals + 1e-8)
        print(f"Feature normalization (0_1_indipendent): min {normalized.min(dim=0).values.tolist()}, max {normalized.max(dim=0).values.tolist()}")

    elif method == "0_1_global":
        # Normalize all features to [0, 1] based on global min and max
        min_val = vertex_features.min()
        max_val = vertex_features.max()
        normalized = (features - min_val) / (max_val - min_val + 1e-8)
        vertex_features_normalized = (vertex_features - min_val) / (max_val - min_val + 1e-8)
        print(f"Feature normalization (0_1_global): min {normalized.min(dim=0).values.tolist()}, max {normalized.max(dim=0).values.tolist()}")

    elif method == "0_center_indipendent":
        # Center each feature channel around 0 and scale to [-1, 1]
        mean_vals = vertex_features.mean(dim=0)
        max_devs = torch.max(torch.abs(vertex_features - mean_vals), dim=0).values
        normalized = (features - mean_vals) / (max_devs + 1e-8)
        vertex_features_normalized = (vertex_features - mean_vals) / (max_devs + 1e-8)
        print(f"Feature normalization (0_center_indipendent): min {normalized.min(dim=0).values.tolist()}, max {normalized.max(dim=0).values.tolist()}")

    elif method == "0_center_global":
        # Center all features around 0 and scale to [-1, 1] based on global max deviation
        mean_val = vertex_features.mean()
        max_dev = torch.max(torch.abs(vertex_features - mean_val))
        normalized = (features - mean_val) / (max_dev + 1e-8)
        vertex_features_normalized = (vertex_features - mean_val) / (max_dev + 1e-8)
        print(f"Feature normalization (0_center_global): min {normalized.min(dim=0).values.tolist()}, max {normalized.max(dim=0).values.tolist()}")

    elif method == "euclidean":
        # Normalize each channel by the Euclidean norm of the vertex features
        norms = torch.norm(vertex_features, dim=0)
        normalized = features / (norms + 1e-8)
        vertex_features_normalized = vertex_features / (norms + 1e-8)
        print(f"Feature normalization (euclidean): min {normalized.min(dim=0).values.tolist()}, max {normalized.max(dim=0).values.tolist()}")

    elif method == "mean_var_vertex":
        mean_vertex_features = vertex_features.mean(dim=0)
        var_vertex_features = vertex_features.std(dim=0)
        features_normalized = (features - mean_vertex_features) / (var_vertex_features + 1e-8)
        vertex_features_normalized = (vertex_features - mean_vertex_features) / (var_vertex_features + 1e-8)
        normalized = features_normalized
        print(f"Feature normalization (mean_var_vertex): min {normalized.min(dim=0).values.tolist()}, max {normalized.max(dim=0).values.tolist()}")

    elif method == "mean_var":
        mean_features = features.mean(dim=0)
        mean_vertex_features = vertex_features.mean(dim=0)
        var_vertex_features = features.std(dim=0)
        features_normalized = (features - mean_features) / (var_vertex_features + 1e-8)
        vertex_features_normalized = (vertex_features - mean_vertex_features) / (var_vertex_features + 1e-8)
        normalized = features_normalized
        print(f"Feature normalization (mean_var_vertex): min {normalized.min(dim=0).values.tolist()}, max {normalized.max(dim=0).values.tolist()}")

    elif method == "mean_var_features":
        mean_features = features.mean(dim=0)
        var_features = features.std(dim=0)
        features_normalized = (features - mean_features) / (var_features + 1e-8)
        vertex_features_normalized = (vertex_features - mean_features) / (var_features + 1e-8)
        normalized = features_normalized
        print(f"Feature normalization (mean_var_features): min {normalized.min(dim=0).values.tolist()}, max {normalized.max(dim=0).values.tolist()}")
    elif method == "diameter":
        vertex_features_normalized = vertex_features / diameter
        normalized = features / diameter
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized, vertex_features_normalized


def initialize_model_and_optimizer(args, device):
    """Initialize the model, optimizer, and loss scaler."""

    # Initialize the model
    if args.method == "FM":
        model = FMCond(
            channels=args.embedding_dim,
            depth=args.depth,
            network=networks.__dict__[args.network](channels=args.embedding_dim),
        )
    elif args.method == "diffusion":
        model = EDMPrecond(
            channels=args.embedding_dim,
            depth=args.depth,
            network=networks.__dict__[args.network](channels=args.embedding_dim),
        )

    model.to(device)

    # Initialize the optimizer and loss scaler (If distributed training different behaviour)
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.learning_rate is None:  # only base_lr is specified
        args.learning_rate = args.blr * eff_batch_size / 128

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )
        model_without_ddp = model.module
        optimizer = torch.optim.AdamW(
            model_without_ddp.parameters(), lr=args.learning_rate
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    loss_scaler = NativeScaler(device)

    print("base lr: %.2e" % (args.learning_rate * 128 / eff_batch_size))
    print("actual lr: %.2e" % args.learning_rate)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    return model, optimizer, loss_scaler


def diffusion_step(model, y, device, args):
    rnd_normal = torch.randn([y.shape[0]], device=y.device)
    t = (rnd_normal * 1.2 - 1).exp()
    weight = (t**2 + 1) / (t**2)

    X_0 = sample_initial_distribution(
        num_points=y.shape[0],
        distribution=args.distribution,
        embedding_dim=args.embedding_dim,
        device=device,
    )

    n = X_0 * t[:, None]
    D_yn = model(y + n, t)
    loss = (weight[:, None] * ((D_yn - y) ** 2)).mean()
    return loss


def fm_step(model, y, path, device, args):
    u = torch.rand(y.shape[0], device=device)
    t = (torch.cos(u * (torch.pi / 2)) ** 2).to(device)

    X_0 = sample_initial_distribution(
        num_points=y.shape[0],
        distribution=args.distribution,
        embedding_dim=args.embedding_dim,
        device=device,
    )

    if path is None:
        raise ValueError("Path object must be provided for FM method.")

    path_sample = path.sample(t=t, x_0=X_0, x_1=y)
    V_y = model(x=path_sample.x_t, sigma=path_sample.t)
    loss = torch.mean((V_y - path_sample.dx_t) ** 2)
    return loss


def train_one_epoch(
    model: torch.nn.Module,
    data_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    step_fn,
    max_norm: float = 0,
    features=None,
    mesh=None,
    embedding_type=None,
    log_writer=None,
    args=None,
):
    """Train the model for one epoch (method-specific logic is injected via step_fn)."""
    model.train(True)

    # Initialize metric logger
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 200

    # Gradient accumulation setup
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if isinstance(data_loader, dict):
        batch_size = data_loader["batch_size"]
        data_loader = range(data_loader["epoch_size"])
        y = features 
    else:
        exit("Data loader must be a dictionary with 'batch_size' and 'epoch_size' keys.")

    # Wrap data loader with logger if verbose
    iterator = (
        metric_logger.log_every(data_loader, print_freq, header)
        if args.verbose else data_loader
    )

    for data_iter_step, batch in enumerate(iterator):
        # Adjust LR during warmup
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        # Prepare input batch
        if isinstance(batch, int):
            ind = np.random.default_rng().choice(y.shape[0], batch_size, replace=True)
            y_batch = y[ind].float().to(device, non_blocking=True)
        else:
            y_batch = batch.to(device, non_blocking=True)

        # Forward + loss
        with torch.amp.autocast(str(device), enabled=False):
            loss = step_fn(model, y_batch, device)

        loss_value = loss.item()

        # Handle invalid loss
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        # Backward pass and optimization
        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # Update metrics
        metric_logger.update(loss=loss_value)

        # Track learning rate
        min_lr, max_lr = 10.0, 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

        # TensorBoard logging
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", max_lr, epoch_1000x)

    # Synchronize metrics across processes
    metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train(args, device):
    data_loader_train = setup_data_loader(args)
    model, optimizer, loss_scaler = initialize_model_and_optimizer(args, device)
    misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)
    
    mesh = trimesh.load(args.data_path, process=False)

    target = args.data_path.split("/")[-1].split(".")[0]
    dists_path = str(Path(args.data_path).parent / f"{target}_dists.npy")
    diameter = get_shape_diameter(mesh, target, dists_path)

    if args.features_path is not None and args.vertex_features_path is not None:
        print(f"Ignoring config_file data_path --> loading features from {args.features_path}")

        ext = os.path.splitext(args.features_path)[1]
        if ext not in [".txt", ".npy"]:
            raise ValueError(f"Features file must be .txt or .npy, got {ext}")
        elif ext == ".txt":
            features = torch.tensor(np.loadtxt(args.features_path).astype(np.float32)).to(device)
            vertex_features = torch.tensor(np.loadtxt(args.vertex_features_path).astype(np.float32)).to(device)
        elif ext == ".npy":
            features = torch.tensor(np.load(args.features_path).astype(np.float32)).to(device)
            vertex_features = torch.tensor(np.loadtxt(args.vertex_features_path).astype(np.float32)).to(device)
        print(f"Loaded features from {args.features_path} | Features shape: {features.shape}")
        print(f"Loaded vertex features from {args.vertex_features_path} | Vertex features shape: {vertex_features.shape}")

        if args.features_interpolation > 0 and mesh is not None:
            features = generate_embeddings(
                mesh=mesh,
                embedding_type=args.embedding_type,
                num_points=args.features_interpolation,
                features=features,
                device=device,
            )
            print(f"Interpolated features over {args.features_interpolation} points | New features shape: {features.shape}")
            np.savetxt(os.path.join(args.output_dir, "features-interpolated.txt"), features.detach().cpu().numpy())
            print(f"Saved interpolated features to {os.path.join(args.output_dir, 'features-interpolated.txt')}")

    else:
        print(f"Computing {args.features_type} features from mesh...")
        vertex_features = compute_features(mesh, args, device)
        print("------------------------------------")
        print(f"vertex_features (shape {list(vertex_features.shape)}):")
        print(f"  min: {vertex_features.min(dim=0).values.tolist()}")
        print(f"  max: {vertex_features.max(dim=0).values.tolist()}")
        print(f"  avg: {vertex_features.mean(dim=0).tolist()}")
        np.savetxt(os.path.join(args.output_dir, "vertex-geodesics.txt"), vertex_features.detach().cpu().numpy())
        print(f"Saved vertex vertex_features to {os.path.join(args.output_dir, 'vertex-geodesics.txt')}")

        # Interpolate the features over the sampled points
        features = generate_embeddings(
            mesh=mesh,
            embedding_type=args.embedding_type,
            num_points=500000,
            features=vertex_features,
            device=device,
        )

        print("------------------------------------")
        print(f"Features interpolated over the sampled points (shape {list(features.shape)}):")
        print(f"  min: {features.min(dim=0).values.tolist()}")
        print(f"  max: {features.max(dim=0).values.tolist()}")
        print(f"  avg: {features.mean(dim=0).tolist()}")
        print("------------------------------------")
        np.savetxt(os.path.join(args.output_dir, "vertex-geodesics-interpolated.txt"), features.detach().cpu().numpy())
        print(f"Saved interpolated features to {os.path.join(args.output_dir, 'vertex-geodesics-interpolated.txt')}")

    if args.features_normalization != "none":
        features, vertex_features = normalize_features(features, vertex_features, args.features_normalization, diameter)
        print("------------------------------------")
        print(f"vertex_features (shape {list(vertex_features.shape)}):")
        print(f"  min: {vertex_features.min(dim=0).values.tolist()}")
        print(f"  max: {vertex_features.max(dim=0).values.tolist()}")
        print(f"  avg: {vertex_features.mean(dim=0).tolist()}")
        print("------------------------------------")
        print(f"Features after '{args.features_normalization}' normalization (shape {list(features.shape)}):")
        print(f"  min: {features.min(dim=0).values.tolist()}")
        print(f"  max: {features.max(dim=0).values.tolist()}")
        print(f"  avg: {features.mean(dim=0).tolist()}")
        np.savetxt(os.path.join(args.output_dir, "vertex-geodesics-interpolated-vnorm.txt"), features.detach().cpu().numpy())
        np.savetxt(os.path.join(args.output_dir, "vertex-geodesics-vnorm.txt"), vertex_features.detach().cpu().numpy())
    else:
        print("Skipping feature normalization")


    logging.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    if args.method == "diffusion":
        step_fn = lambda model, y, device: diffusion_step(model, y, device, args)
    elif args.method == "FM":
        path = AffineProbPath(scheduler=CondOTScheduler())
        step_fn = lambda model, y, device: fm_step(model, y, path, device, args)
    else:
        raise ValueError(f"Unknown method {args.method}")

    for epoch in trange(args.start_epoch, args.epochs, desc="Epochs"):
        train_stats = train_one_epoch(
            model=model,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            step_fn=step_fn,
            max_norm=args.clip_grad,
            features=features,
            mesh=mesh,
            embedding_type=args.embedding_type,
            args=args,
        )

        if args.output_dir and (epoch % 100 == 0 or epoch + 1 == args.epochs):
            if epoch + 1 == args.epochs:
                if args.distributed:
                    misc.save_model(
                        args=args,
                        model=model,
                        model_without_ddp=model.module,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        epoch=epoch,
                    )
                else:
                    misc.save_model(
                        args=args,
                        model=model,
                        model_without_ddp=model,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        epoch=epoch,
                    )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir and misc.is_main_process():
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    logging.info(
        f"Training completed in {str(datetime.timedelta(seconds=int(total_time)))}"
    )


def inference(args, device):
    if args.method == "FM":
        model = FMCond(
            channels=args.embedding_dim,
            depth=args.depth,
            network=models.__dict__[args.network](channels=args.embedding_dim),
        )
    elif args.method == "diffusion":
        model = EDMPrecond(
            channels=args.embedding_dim,
            depth=args.depth,
            network=models.__dict__[args.network](channels=args.embedding_dim),
        )
    model.to(device)
    model.load_state_dict(
        torch.load(
            args.output_dir + "/checkpoint-" + str(args.epochs - 1) + ".pth",
            map_location=device,
            weights_only=False,
        )["model"],
        strict=True,
    )

    noise = sample_initial_distribution(
        num_points=args.num_points_inference,
        distribution=args.distribution,
        embedding_dim=args.embedding_dim,
        device=device,
    )

    with torch.no_grad():
        sample, _ = model.sample(
            noise=noise, num_steps=args.num_steps, intermediate=True
        )

    # Save the generated samples
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        np.save(
            os.path.join(args.output_dir, "generated_samples.npy"), sample.cpu().numpy()
        )
    else:
        np.save("generated_samples.npy", sample.cpu().numpy())


def main():
    args = get_inline_arg()

    # Load arguments from a JSON file if specified
    if args.config:
        with open(args.config, "r") as f:
            config_args = json.load(f)
        for key, value in config_args.items():
            if not hasattr(args, key):
                continue
            setattr(args, key, value)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        setup_logging(args.output_dir)

    device = initialize_device_and_seed(args)

    if args.train:
        train(args, device)

    if args.inference:
        inference(args, device)


if __name__ == "__main__":
    main()
