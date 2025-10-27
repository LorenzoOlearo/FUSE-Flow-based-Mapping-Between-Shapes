import numpy as np
import torch
import trimesh
import os
import sys
import argparse
import json

from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import Mapping, Any


from typing import List, Tuple, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from matching_experiments import *

@dataclass
class SkinningConfig:
    kinect_landmarks: List[int]
    smplx_landmarks: List[int]
    smplx_flows_path: Path
    smplx_mesh_path: Path
    smplx_features_path: Path
    smplx_dists_path: Path
    smplx_dataset_extension: str
    kinect_corr_path: Path
    kinect_dataset_path: Path
    kinect_dataset_extension: str
    kinect_corres_path: Path
    kinect_flows_path: Path
    kinect_features_path: Path
    kinect_dists_path: Path
    output_path: Path

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> "SkinningConfig":
        return cls(
            kinect_landmarks=m["kinect_landmarks"],
            smplx_landmarks=m["smplx_landmarks"],
            smplx_flows_path=Path(m["smplx_flows_path"]),
            smplx_mesh_path=Path(m["smplx_mesh_path"]),
            smplx_features_path=Path(m["smplx_features_path"]),
            smplx_dists_path=Path(m["smplx_dists_path"]),
            smplx_dataset_extension=m["smplx_dataset_extension"],
            kinect_dataset_path=Path(m["kinect_dataset_path"]),
            kinect_corr_path=Path(m["kinect_corr_path"]),
            kinect_dataset_extension=m["kinect_dataset_extension"],
            kinect_corres_path=Path(m["kinect_corres_path"]),
            kinect_flows_path=Path(m["kinect_flows_path"]),
            kinect_features_path=Path(m["kinect_features_path"]),
            kinect_dists_path=Path(m["kinect_dists_path"]),
            output_path=Path(m["output_path"]),
        )
        
        
def process_smplx_element(
    element: str,
    skinning_config: SkinningConfig,
    device: torch.device,
):
    model = FMCond(
        channels=int(5),
        network=MLP(channels=int(5)).to(device),
    )
    
    landmarks = skinning_config.smplx_landmarks
    
    mesh = trimesh.load(
        skinning_config.smplx_mesh_path,
        process=False,
    )
    
    features_path = Path(skinning_config.smplx_features_path, element, f"vertex-geodesics-vnorm.txt")
    features = torch.tensor(np.loadtxt(features_path).astype(np.float32)).to(device)
   
    smplx_flows_path = Path(skinning_config.smplx_flows_path, element, "checkpoint-9999.pth") 
    model.load_state_dict(
        torch.load(
            Path(smplx_flows_path),
            weights_only=False,
        )["model"],
        strict=True,
    )
    model.eval()
    model.to(device)
    
    return mesh, landmarks, features, model


def process_kinect_element(
    element: str,
    skinning_config: SkinningConfig,
    device: torch.device,
):
    model = FMCond(
        channels=int(5),
        network=MLP(channels=int(5)).to(device),
    )
    
    pt = trimesh.load(
        Path(skinning_config.kinect_dataset_path, element + skinning_config.kinect_dataset_extension),
        process=False,
    )
    corr = np.loadtxt(Path(skinning_config.kinect_corr_path, element + ".vts")).astype(int)
    landmarks = corr[skinning_config.kinect_landmarks]
   
    features_path = Path(skinning_config.kinect_features_path, element, f"vertex-geodesics-vnorm.txt") 
    tqdm.write(f"Loading precomputed features for {element} from {features_path}")
    features = np.loadtxt(features_path).astype(np.float32)
    features = torch.tensor(features).to(device)
    
    model.load_state_dict(
        torch.load(
            Path(skinning_config.kinect_flows_path, element, "checkpoint-9999.pth"),
            weights_only=False,
        )["model"],
        strict=True,
    )
    
    model.eval()
    model.to(device)
    
    return pt, landmarks, features, model


def plot_results(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    source: str,
    target: str,
    results: Dict[str, torch.Tensor],
    output_dir: str,
    plot_html: bool,
    plot_png: bool,
    max_points: int = 100_000,
) -> None:
    """Plot correspondences for each method."""
    source_points = source_points[:max_points]
   
    for method, p2p in results.items():
        start_end_subplot(
            source_points,
            target_points[p2p],
            plots_path=str(output_dir),
            run_name=f"matching-{method}-{source}-{target}",
            show=False,
            html=plot_html,
            png=plot_png,
        )

        
def process_pair(
    source: str,
    target: str,
    skinning_config: SkinningConfig,
    device: torch.device,
):
    pt, kinect_landmarks, kinect_features, kinect_model = process_kinect_element(
        element=source,
        skinning_config=skinning_config,
        device=device,
    )
    
    mesh, smplx_landmarks, smplx_features, smplx_model = process_smplx_element(
        element=target,
        skinning_config=skinning_config,
        device=device,
    )
    
    p2p_knn, elapsed_knn = compute_p2p_with_knn(kinect_features, smplx_features)
    p2p_flow, elapsed_flow = compute_p2p_with_flows_composition(kinect_features, smplx_features, kinect_model, smplx_model, device)
    p2p_ndp, elapsed_ndp = compute_p2p_with_ndp_sdf(source_vertex=pt.vertices, target_vertex=mesh.vertices, source_landmarks=kinect_landmarks, target_landmarks=smplx_landmarks)
    
    results = {
        "knn": p2p_knn,
        "flow": p2p_flow,
        "ndp": p2p_ndp,
    }

    for method, p2p in results.items():
        tqdm.write(f"{method} unique: {len(np.unique(p2p))} / {len(p2p)}")
        output_path = Path(
            skinning_config.output_path,
            "p2p",
            f"p2p-{method}-{source}-to-{target}",
        )
        os.makedirs(output_path.parent, exist_ok=True)
        np.save(output_path, p2p.astype(int))
        tqdm.write(f"Saved {method} p2p to {output_path}")
        
    
    os.makedirs(skinning_config.output_path, exist_ok=True) 
    plot_results(
        source_points=torch.tensor(pt.vertices.astype(np.float32)).to(device),
        target_points=torch.tensor(mesh.vertices.astype(np.float32)).to(device),
        results=results,
        source=source,
        target=target,
        output_dir=str(skinning_config.output_path),
        plot_html=True,
        plot_png=True,
    )


def main(args):
    with open(args.config, "r") as f:
        config = json.load(f)

    device = config["device"]
    torch.cuda.set_device(device)

    skinning_config = SkinningConfig.from_mapping(config["skinning_config"])
    tqdm.write("Skinning Config:")
    for key, value in skinning_config.__dict__.items():
        tqdm.write(f"  {key}: {value}")

    sources = get_targets_kinect(Path(config["skinning_config"]["kinect_flows_path"]))
    targets = get_targets_smplx(Path(config["skinning_config"]["smplx_flows_path"]))
    pairs = [(s, t) for s in sources for t in targets]
    tqdm.write(f"Processing {len(pairs)} source-target pairs.")
    
    for source, target in tqdm(pairs, desc="Processing shape pairs", unit="pair", dynamic_ncols=True):
        tqdm.write(f"Processing {source} -> {target}")
        process_pair(
            source=source,
            target=target,
            skinning_config=skinning_config,
            device=device,
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Match the kinect shapes to the SMPLX template mesh"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the config file",
        required=True,
        default="config.json",
    )


    args = parser.parse_args()

    tqdm.write("------------------------------------------")
    for arg in vars(args):
        tqdm.write(f"{arg}: {getattr(args, arg)}")
    tqdm.write("------------------------------------------")


    main(args)
