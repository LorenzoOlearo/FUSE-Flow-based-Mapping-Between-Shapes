import numpy as np
import torch
import trimesh

def normalize_mesh(mesh):
    rescale = max(mesh.extents) / 2.
    tform = [
        -(mesh.bounds[1][i] + mesh.bounds[0][i]) / 2.
        for i in range(3)
    ]
    matrix = np.eye(4)
    matrix[:3, 3] = tform
    mesh.apply_transform(matrix)
    matrix = np.eye(4)
    matrix[:3, :3] /= rescale
    mesh.apply_transform(matrix)
    
    return mesh


def sample_sphere_volume(radius, center, num_points,device='cuda'):
    center = torch.tensor(center, dtype=torch.float32,device=device)
    r = radius * torch.rand(num_points,device=device)    ** (1/3)   # Sample radius with proper scaling for uniform distribution
    theta = torch.rand(num_points,device=device)    * 2 * np.pi           # Azimuthal angle
    phi = torch.acos(1 - 2 * torch.rand(num_points,device=device)   )     # Polar angle

    # Convert to Cartesian coordinates
    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)

    points = torch.stack((x, y, z), dim=1) + center

    return points




def generate_samples(args, device='cuda'):
    
    if args.distribution == 'Gaussian':
        if args.embedding == 'landmark_feat':
            noise = torch.randn(args.num_points_inference, 3+len(args.landmarks)).to(device)
        else:
            noise = torch.randn(args.num_points_inference, 3).to(device)

    elif args.distribution == 'Sphere':
        noise = sample_sphere_volume(
            radius=1,
            center=(0, 0, 0),
            num_points=args.num_points_inference,
            device=device
        )

    elif args.distribution == 'Gaussian_Optimized':
        mesh = normalize_mesh(trimesh.load(args.data_path))
        n = torch.randn(args.num_points_inference, 3).to(device)
        n += torch.tensor(mesh.vertices[:, :3], device=device).mean(dim=0)
        n *= torch.tensor(mesh.vertices[:, :3], device=device).std(dim=0)
        noise = n

    return noise


# Add this to a new cell in your notebook
import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import coo_matrix

def compute_geodesic_distances(mesh, source_index):
    """
    Compute geodesic distances from a source vertex to all other vertices.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        The mesh on which to compute geodesic distances
    source_index : int
        Index of the source vertex
        
    Returns:
    --------
    distances : np.ndarray
        Array of distances from source to all vertices
    """
    # Get unique edges
    edges = mesh.edges_unique
    
    # Calculate edge lengths
    edge_lengths = np.linalg.norm(
        mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], 
        axis=1
    )
    
    # Create a sparse adjacency matrix
    n_vertices = len(mesh.vertices)
    graph = coo_matrix(
        (edge_lengths, (edges[:, 0], edges[:, 1])),
        shape=(n_vertices, n_vertices)
    )
    
    # Make the graph symmetric (undirected)
    graph = graph + graph.T
    
    # Compute shortest paths
    distances, predecessors = shortest_path(
        csgraph=graph, 
        directed=False,
        indices=source_index,
        return_predecessors=True
    )
    
    return distances



def get_ldmk_feats(mesh, num_points, lm,device=torch.device('cuda:0')):
    """
    Compute approximate geodesic distances between sampled points and landmarks.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        The input mesh
    num_points : int
        Number of points to sample from the mesh surface
    lm : numpy.ndarray
        Array of landmark vertex indices
    
    Returns:
    --------
    torch.Tensor
        Tensor of shape (num_points, len(lm)) containing geodesic distances 
        from each sampled point to each landmark
    """
    # 1) Sample points from the mesh surface
    samples, _ = trimesh.sample.sample_surface(mesh, num_points)
    samples = samples.astype(np.float32)
    
    # 2) Find the nearest mesh vertices for each sampled point
    samples_tensor = torch.tensor(samples, device=device).to(torch.float32).to(device)
    mesh_vertices_tensor = torch.tensor(mesh.vertices, device=device).to(torch.float32).to(device)
    distances = torch.cdist(samples_tensor, mesh_vertices_tensor)
    vertex_indices = torch.argmin(distances, dim=1).cpu().numpy()
    
    # Calculate Euclidean distances between sampled points and their nearest vertices
    mesh_vertices = mesh_vertices_tensor[vertex_indices]
    point_to_vertex_distances = torch.linalg.norm(samples_tensor - mesh_vertices, axis=1)
    
    # 3) Compute geodesic distances from each vertex to each landmark
    # This gives a matrix of shape (num_vertices, len(lm))
    landmark_geodesic_distances = torch.tensor(compute_geodesic_distances(mesh, lm).T).to(device).to(torch.float32) 
    
    # Get the geodesic distances for the nearest vertices to our sampled points
    # This gives a matrix of shape (num_points, len(lm))
    vertex_to_landmark_distances = landmark_geodesic_distances[vertex_indices]
    
    # 4) Add the point-to-vertex distances to get approximate geodesic distances
    # from sampled points to landmarks
    # Expand point_to_vertex_distances to add to each landmark distance
    point_to_vertex_distances_expanded = point_to_vertex_distances[:,None]
    geodesic_distances = vertex_to_landmark_distances + point_to_vertex_distances_expanded
    
    return torch.cat([samples_tensor,geodesic_distances],-1)




def generate_embeddings(mesh, args, device='cuda'):
    
    if args.embedding =='landmark_feat':
        embedding = get_ldmk_feats(mesh, args.num_points_train, args.landmarks, device=device)
    elif args.embedding == 'xyz':
        if len(mesh.faces)>0:
            samples, _ = trimesh.sample.sample_surface(mesh, args.num_points_train)
        else:
            samples = mesh.vertices
        
        embedding= torch.tensor(samples, dtype=torch.float32, device=device)
    
    return embedding