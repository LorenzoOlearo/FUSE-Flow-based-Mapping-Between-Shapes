import numpy as np
import os
import logging
import random
import torch
import torch.backends.cudnn as cudnn

from util import lr_decay as lrd, misc



def setup_logging(output_dir: str):
    """Set up logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, "training.log"), mode="w", encoding="utf-8")
        ]
    )


def initialize_device_and_seed(args):
    """Initialize device and set random seeds for reproducible results."""
    misc.init_distributed_mode(args)
    device = torch.device(args.device)

    # Seed setup
    seed = int(args.seed) + misc.get_rank()
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic backend settings
    cudnn.benchmark = False
    cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

    # Disable TF32 for full precision reproducibility
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print(f"[Seed Init] Using seed {seed} on device {device}")
    return device
