"""
Main script for training and inference of the neural network model.
This script includes functions for training the model, performing inference, and setting up the data loader.
It also includes command-line argument parsing and logging setup.
"""

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import logging
import torch
from util import lr_decay as  misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from util.train_utils import *
import trimesh

import math
import sys

import util.misc as misc
import util.lr_sched as lr_sched

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath

from model import models,networks
from model.models import *
from model.networks import *

from model.losses import chamfer_dist, hausdorff_dist

from util.mesh_utils import *
from util.plot import *


def get_inline_arg():
    parser = argparse.ArgumentParser('Train', add_help=False)
    # common parameters
    parser.add_argument('--config', default=None, type=str, help='Path to the config file')    # Model parameters
    parser.add_argument('--method', default='FM', type=str, help='Method used for training')
    parser.add_argument('--network', default='MLP', type=str, help='Network used for training')
    parser.add_argument("--train", action="store_true", help="Train a <run_name> model")
    parser.add_argument("--inference", action="store_true", help="Perform inference on the <run_name> model")
    parser.add_argument('--device', default='cuda:1',
                        help='device to use for training / testing')
    parser.add_argument('--run_name', default='RUN',
                        help='Name of the run')
    parser.add_argument('--embedding', default='xyz', type=str)
    parser.add_argument('--landmarks', default=[], type=list)
    parser.add_argument('--seed', default=21, type=int)
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--learning_rate', type=float, default=0.01, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--batch_size', default=10000,type=int,help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus') #1024*64*2

    parser.add_argument('--num_points_inference', default=10000, type=int, help='Number of points for inference')
    parser.add_argument('--num_points_train', default=10000, type=int, help='Number of points for inference')#2048 * 64 * 4 * 64

    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--num-steps', default=64, type=int)
    parser.add_argument('--model', default='FMCond', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--depth', default=6, type=int, metavar='MODEL')
    
    parser.add_argument('--data_path', default='shapes/Jellyfish_lamp_part_A__B_normalized.obj', type=str,
                        help='dataset path')

    parser.add_argument('--distribution', default='Gaussian', type=str, )

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--blr', type=float, default=5e-7, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=5e-7, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--intermediate', action='store_true')
    parser.add_argument('--output_dir', default='./output/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output/',
                        help='path where to tensorboard log')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=32, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=2, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=2, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    args = parser.parse_args()

    return args



def setup_data_loader(args):
    """Set up the data loader based on the input data path."""
    if any(primitive in args.data_path for primitive in ['sphere', 'plane', 'volume']):
        data_loader_train = {
            'obj_file': None,
            'primitive': args.data_path,
            'batch_size': args.batch_size,
            'epoch_size': args.num_points_train // args.batch_size, 
        }
    else:
        data_loader_train = {
            'obj_file': args.data_path,
            'batch_size': args.batch_size,
            'epoch_size': args.num_points_train // args.batch_size,
        }
    return data_loader_train



def initialize_model_and_optimizer(args,device):
    """Initialize the model, optimizer, and loss scaler."""
    
    # Initialize the model
    model = models.__dict__[args.model](channels=3, depth=args.depth,network=networks.__dict__[args.network](channels=3))
    model.to(device)
    
    
    # Initialize the optimizer and loss scaler (If distributed training different behaviour)
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.learning_rate is None:  # only base_lr is specified
        args.learning_rate = args.blr * eff_batch_size / 128
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp =model.module
        optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.learning_rate)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    loss_scaler = NativeScaler()
    
    
    # Log
    print("base lr: %.2e" % (args.learning_rate * 128 / eff_batch_size))
    print("actual lr: %.2e" % args.learning_rate)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    return model, optimizer, loss_scaler


def train_one_epoch(model: torch.nn.Module,
                    data_loader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, args=None,mesh=None, path=None):

    model.train(True)

    # Initialize metric logger
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 20

    # Gradient accumulation setup
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    # Log
    if log_writer is not None:
        print(f'log_dir: {log_writer.log_dir}')
    print(data_loader)


    if isinstance(data_loader, dict):
        batch_size = data_loader['batch_size']
        if len(mesh.faces)>0:
            samples, _ = trimesh.sample.sample_surface(mesh, args.num_points_train)
        else:
            samples = mesh.vertices

        samples = samples.astype(np.float32)
        data_loader = range(data_loader['epoch_size'])

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        # Prepare input data
        if isinstance(batch, int):
            ind = np.random.default_rng().choice(samples.shape[0], batch_size, replace=True)
            y = samples[ind]
            y = torch.from_numpy(y).float().to(device, non_blocking=True)
        else:
            y = batch.to(device, non_blocking=True)

        # Forward pass and loss computation
        with torch.amp.autocast(args.device,enabled=False):
            
            if args.method == 'Geomdist':
                rnd_normal = torch.randn([y.shape[0],], device=y.device)
                t = (rnd_normal * 1.2 -1).exp()
                weight = (t ** 2 + 1) / (t) ** 2

            else:
                #t= torch.rand(y.shape[0], device=device)   # random sampling
                u = torch.rand(y.shape[0], device=device)
                t = (torch.cos(u * (torch.pi / 2)) ** 2).to(device)

            X_0 = generate_samples(args, device=device)

            if args.method=='Geomdist':
                #geomdist training logic
                n=X_0*t[:,None]
                D_yn = model(y + n, t)
                loss = ( weight[:, None] * ((D_yn - y) ** 2) ).mean()
            else:
                #flowmatching training logic
                path_sample = path.sample(t=t, x_0=X_0, x_1=y)

                V_y = model(x=path_sample.x_t, sigma=path_sample.t)
                loss = torch.mean((V_y - path_sample.dx_t)**2)

        loss_value = loss.item()

        # Handle invalid loss values
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        # Backward pass and gradient update
        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # Update metrics
        metric_logger.update(loss=loss_value)

        # Track learning rate
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        # Log metrics to TensorBoard
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # Synchronize metrics across processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # Return averaged metrics
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def train(args, device):
    """Main training loop."""
    # Set up the data loader
    data_loader_train = setup_data_loader(args)
    # Initialize the model, optimizer, and loss scaler
    model, optimizer, loss_scaler = initialize_model_and_optimizer(args, device)
    misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)    
    #load the data
    mesh= normalize_mesh(trimesh.load(args.data_path, process=False))  
    
    logging.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    if args.method=='FM':
        #define the pathSampler
        path= AffineProbPath(scheduler=CondOTScheduler())
    else: 
        path=None

    for epoch in range(args.start_epoch, args.epochs):
        # Train for one epoch
        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=None, args=args,mesh=mesh, path=path
        )

        # Save checkpoints
        if args.output_dir and (epoch % 100 == 0 or epoch + 1 == args.epochs):
            #save model
            if epoch + 1 == args.epochs:
                if args.distributed:
                    misc.save_model(args=args,model=model, model_without_ddp=model.module,optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)
                else:
                    misc.save_model(args=args, model=model,model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

            #save chamfer and hausdoorff
            noise= generate_samples(args, device=device)
            model.eval()
            with torch.no_grad():
                sample = model.sample(noise=noise, num_steps=args.num_steps)
            with open(os.path.join(args.output_dir, "chamfer_distance.txt"), mode="a", encoding="utf-8") as f:
                f.write(f"Epoch {epoch}: Chamfer distance: {chamfer_dist(sample[:,:3].unsqueeze(0).to(torch.double), torch.tensor(mesh.vertices).unsqueeze(0).to(device)).item()}\n")
            with open(os.path.join(args.output_dir, "hausdorff_distance.txt"), mode="a", encoding="utf-8") as f:
                f.write(f"Epoch {epoch}: Hausdorff distance: {hausdorff_dist(sample[:,:3].unsqueeze(0).to(torch.double), torch.tensor(mesh.vertices).unsqueeze(0).to(device)).item()}\n")
        
        #Log stats
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        
        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        # Train for one epoch

    total_time = time.time() - start_time
    logging.info(f"Training completed in {str(datetime.timedelta(seconds=int(total_time)))}")



def inference(args, device):

    model = models.__dict__[args.model](channels=3, depth=args.depth,network=models.__dict__[args.network]())   
    model.to(device)
    model.load_state_dict(torch.load(args.output_dir + '/checkpoint-'+str(args.epochs-1)+'.pth', map_location=device,weights_only=False)['model'], strict=True)
    
    noise = generate_samples(args, device=device)

    with torch.no_grad():
        sample,_ = model.sample(noise=noise, num_steps=args.num_steps, intermediate=True)
    
    start_end_subplot(noise.cpu(), sample.cpu(),plots_path=args.output_dir,run_name=args.run_name)




def main():
    args = get_inline_arg()
    
    # Load arguments from a JSON file if specified
    if args.config:
        with open(args.config, 'r') as f:
            config_args = json.load(f)
        for key, value in config_args.items():
            if not hasattr(args, key):
                raise ValueError(f"Unknown argument in config file: {key}")
            setattr(args, key, value)

    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        setup_logging(args.output_dir)
   
    device = initialize_device_and_seed(args)
    
    if args.train:
        train(args, device)

    if args.inference:
        inference(args, device)



if __name__ == '__main__':
    main()
