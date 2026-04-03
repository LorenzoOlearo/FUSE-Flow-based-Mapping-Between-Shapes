import os
from pathlib import Path
from typing import List

from tqdm import tqdm


def is_in_range(name):
    try:
        num = int("".join(filter(str.isdigit, name)))
        return 80 <= num <= 100
    except ValueError:
        return False


def get_targets_faust(args) -> List[str]:
    """Determine which targets to process."""
    targets = []
    for i in range(80, 100):
        if (
            args.source_rep == "sdf"
            or args.target_rep == "sdf"
            or args.source_rep == "pt"
            or args.target_rep == "pt"
        ):
            if i == 86 or i == 87 or i == 96 or i == 97:
                tqdm.write(f"Skipping topologically incorrect shape tr_reg_{i:03d}")
                continue
        target = f"tr_reg_{i:03d}"
        targets.append(target)

    tqdm.write(f"Processing all targets: {targets}")
    return targets


def get_targets_smal(flows_path) -> List[str]:
    """Determine which targets to process."""
    targets = [
        f.name
        for f in flows_path.iterdir()
        if f.is_dir()
        and f.name.startswith(("cougar", "hippo", "horse"))
        and (flows_path / f.name / "checkpoint-best.pth").is_file()
    ]

    tqdm.write(f"Processing targets: {targets}")
    return targets


def get_targets_kinect(flows_path) -> List[str]:
    """Determine which targets to process."""
    targets = [
        f.name
        for f in flows_path.iterdir()
        if f.is_dir() and f.name.startswith(("data"))
    ]

    tqdm.write(f"Processing all targets: {targets}")
    return targets


def get_targets_surreal(flows_path) -> List[str]:
    targets = [
        f.name
        for f in flows_path.iterdir()
        if f.is_dir() and f.name.startswith(("surreal"))
    ]
    return targets


def get_targets_smplx(flows_path) -> List[str]:
    targets = [
        f.name
        for f in flows_path.iterdir()
        if f.is_dir() and f.name.startswith(("SMPLX"))
    ]
    return targets


def get_targets_shrec20(flows_path) -> List[str]:
    """Determine which targets to process."""
    targets = [
        f.name
        for f in flows_path.iterdir()
        if f.is_dir() and (flows_path / f.name / "checkpoint-9999.pth").is_file()
    ]

    tqdm.write(f"Processing targets: {targets}")
    return targets


def get_targets_shrec19(flows_path) -> List[str]:
    """Determine which targets to process."""
    targets = [
        f.name
        for f in flows_path.iterdir()
        if f.is_dir() and (flows_path / f.name / "checkpoint-9999.pth").is_file()
    ]

    tqdm.write(f"Processing targets: {targets}")
    return targets


def get_targets_tosca(flows_path) -> List[str]:
    """Determine which targets to process."""
    targets = [
        f.name
        for f in flows_path.iterdir()
        if f.is_dir()
        and f.name.startswith("cat")
        and (flows_path / f.name / "checkpoint-9999.pth").is_file()
    ]

    tqdm.write(f"Processing targets: {targets}")
    return targets


def get_targets_faust_r(args) -> List[str]:
    """Determine which targets to process."""
    targets = []
    for i in range(80, 100):
        target = f"tr_reg_{i:03d}"
        targets.append(target)

    tqdm.write(f"Processing all targets: {targets}")
    return targets


def get_targets_topkids(args) -> List[str]:
    """Determine which targets to process."""
    targets = []
    base_name = "kid"
    for i in range(0, 26):
        target = f"{base_name}{i:02d}"
        targets.append(target)

    tqdm.write(f"Processing all targets: {targets}")
    return targets
