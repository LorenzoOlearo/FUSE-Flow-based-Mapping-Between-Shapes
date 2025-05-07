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
        noise = torch.randn(args.num_points_inference, args.embedding_dim).to(device)

    
    elif args.distribution == 'Sphere':
        if args.embedding_dim==3:
            
            noise = sample_sphere_volume(
                radius=1,
                center=(0, 0, 0),
                num_points=args.num_points_inference,
                device=device
            )
        else:
            raise NotImplementedError("Sampling from a sphere with dimensions other than 3 is not implemented.")

    #elif args.distribution == 'Gaussian_Optimized':
    #    mesh = normalize_mesh(trimesh.load(args.data_path))
    #    n = torch.randn(args.num_points_inference, 3).to(device)
    #    n += torch.tensor(mesh.vertices[:, :3], device=device).mean(dim=0)
    #    n *= torch.tensor(mesh.vertices[:, :3], device=device).std(dim=0)
    #    noise = n

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



def get_interpolated_feats(mesh, args, device=torch.device('cuda:0')):
    """
    Compute approximate geodesic distances between sampled points and landmarks.
    Fully vectorized implementation without any point-by-point loops.
    """
    print('interpolating features')
    # Sample points
    samples, face_indices = trimesh.sample.sample_surface(mesh, args.num_points_train)
    samples = samples.astype(np.float32)
    
    # Convert to tensors
    samples_tensor = torch.tensor(samples, device=device).float()
    faces = mesh.faces[face_indices]
    
    # Get all vertices for each face (batch processing)
    face_vertices = mesh.vertices[faces]  # Shape: [num_points, 3, 3]
    
    # Compute face normals (vectorized)
    v0 = face_vertices[:, 1] - face_vertices[:, 0]  # Shape: [num_points, 3]
    v1 = face_vertices[:, 2] - face_vertices[:, 0]  # Shape: [num_points, 3]
    
    
    # Calculate vectors for barycentric coordinates
    v2 = samples - face_vertices[:, 0]  # Shape: [num_points, 3]
    
    # Compute dot products for barycentric calculation
    d00 = np.sum(v0 * v0, axis=1)  # Shape: [num_points]
    d01 = np.sum(v0 * v1, axis=1)  # Shape: [num_points]
    d11 = np.sum(v1 * v1, axis=1)  # Shape: [num_points]
    d20 = np.sum(v2 * v0, axis=1)  # Shape: [num_points]
    d21 = np.sum(v2 * v1, axis=1)  # Shape: [num_points]
    
    # Calculate denominator
    denom = d00 * d11 - d01 * d01  # Shape: [num_points]
    
    # Handle barycentric coordinates calculation
    valid_mask = np.abs(denom) >= 1e-10
    
    # Initialize barycentric coordinates
    v = np.zeros(len(samples))
    w = np.zeros(len(samples))
    
    # Calculate barycentric coordinates for valid triangles
    v[valid_mask] = (d11[valid_mask] * d20[valid_mask] - d01[valid_mask] * d21[valid_mask]) / denom[valid_mask]
    w[valid_mask] = (d00[valid_mask] * d21[valid_mask] - d01[valid_mask] * d20[valid_mask]) / denom[valid_mask]
    u = 1.0 - v - w
    
    # Clamp and normalize barycentric coordinates
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    w = np.clip(w, 0.0, 1.0)
    
    total = u + v + w
    u = u / total
    v = v / total
    w = w / total
    
    # Convert everything to PyTorch for faster GPU computation
    u_tensor = torch.tensor(u, device=device).float().unsqueeze(1)  # Shape: [num_points, 1]
    v_tensor = torch.tensor(v, device=device).float().unsqueeze(1)  # Shape: [num_points, 1]
    w_tensor = torch.tensor(w, device=device).float().unsqueeze(1)  # Shape: [num_points, 1]
    valid_mask_tensor = torch.tensor(valid_mask, device=device)
    
    # Convert face indices to PyTorch tensors
    face_idx_tensor = torch.tensor(faces, device=device)  # Shape: [num_points, 3]
    
    # Get landmark geodesic distances for all vertices in all faces
    # Shape: [num_points, 3, num_landmarks]
    vertex_feats = args.features[face_idx_tensor]
    
    # Interpolate using barycentric coordinates
    # Multiply each vertex's geodesic distances by its barycentric weight
    # Shape after multiplication: [num_points, num_landmarks]
    interpolated_feats = (
        u_tensor * vertex_feats[:, 0] +  # First vertex contribution
        v_tensor * vertex_feats[:, 1] +  # Second vertex contribution
        w_tensor * vertex_feats[:, 2]    # Third vertex contribution
    )
    
    # For invalid triangles, use nearest vertex
    if not torch.all(valid_mask_tensor):
        # Calculate squared distances to each vertex for all points
        # Shape: [num_points, 3]
        vertex_distances = torch.sum((torch.tensor(face_vertices, device=device) - 
                                        samples_tensor.unsqueeze(1))**2, dim=2)
        
        # Find nearest vertex index (0, 1, or 2) for each face
        # Shape: [num_points]
        nearest_vertex_idx = torch.argmin(vertex_distances, dim=1)
        
        # Create indices for gathering
        point_indices = torch.arange(len(samples), device=device)
        
        # Get the actual vertex indices from the faces
        # Shape: [num_points]
        selected_vertices = face_idx_tensor[point_indices, nearest_vertex_idx]
        
        # Get geodesic distances for nearest vertices
        # Shape: [num_points, num_landmarks]
        nearest_geodesic = args.features[selected_vertices]
        
        # Use nearest vertex distances for invalid triangles
        invalid_mask_tensor = ~valid_mask_tensor
        interpolated_feats[invalid_mask_tensor] = nearest_geodesic[invalid_mask_tensor]
    
    return torch.cat([samples_tensor, interpolated_feats], -1)

def get_ldmk_feats_precise(mesh, args, device=torch.device('cuda:0')):
    """
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
    samples, face_indices = trimesh.sample.sample_surface(mesh, args.num_points_train)
    sampled_faces = torch.tensor( mesh.faces[face_indices]).to(device) # Shape: [num_points, 3]
    samples = torch.tensor(samples.astype(np.float32)).to(torch.float32).to(device)
    # 2) Find the nearest vertex of the face of each sampled point to each landmark
    vertices = torch.tensor(mesh.vertices, device=device).to(torch.float32)

    # Get the vertices of the faces corresponding to each sampled point
    face_vertices = vertices[sampled_faces]  # Shape: [num_faces, 3, 3]
    face_landmark_dist = args.features[sampled_faces].to(torch.float32).to(device)
    # Compute Euclidean distances from each sampled point to the vertices of its corresponding face
    sampled_point_expanded = samples.unsqueeze(1)  # Shape: [num_points, 1, 3]
    face_vertex_dist = torch.linalg.norm(face_vertices - sampled_point_expanded, dim=2)  # Shape: [num_points, 3]
    
    geodesic_distances =  torch.min(face_landmark_dist,dim=1)+face_vertex_dist[:,torch.argmin(face_landmark_dist,dim=1)]

    return torch.cat([samples,geodesic_distances],-1)        
    

def generate_embeddings(mesh, args, device='cuda'):
    
    if len(mesh.faces) > 0:
        if args.embedding =='features':
            
            embedding = get_interpolated_feats(mesh, args, device=device)
        #elif args.embedding =='landmark_feat_mix':
        #    embedding = get_ldmk_feats(mesh, args, device=device)
        #    samples = mesh.vertices
        #    samples_tensor = torch.tensor(samples, device=device).float()
        #    embedding_verts=torch.cat([samples_tensor, args.ldmk_geodesic_distances], -1)
        #    embedding = torch.cat([embedding, embedding_verts], dim=0)
    
        elif args.embedding == 'xyz':
            samples, _ = trimesh.sample.sample_surface(mesh, args.num_points_train)
            embedding= torch.tensor(samples, dtype=torch.float32, device=device)

    else:
        if args.embedding == 'xyz':
            samples = mesh.vertices
            embedding= torch.tensor(samples, dtype=torch.float32, device=device)

        elif args.embedding == 'features':
            samples = mesh.vertices
            samples_tensor = torch.tensor(samples, device=device).float()
            embedding_verts=torch.cat([samples_tensor, args.features], -1)
            embedding = torch.cat([embedding, embedding_verts], dim=0)

    return embedding


def compute_features(mesh,args,device='cuda'):
    """
    Compute geodesic distances from each vertex to the landmarks.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        The input mesh
    lm : numpy.ndarray
        Array of landmark vertex indices

    Returns:
    --------
    torch.Tensor
        Tensor of shape (num_vertices, len(lm)) containing geodesic distances 
        from each vertex to each landmark
    """
    
    # Compute geodesic distances for all vertices in the mesh
    if args.embedding == 'xyz':
        return None
    if args.features_type == 'landmarks':
        # Get the landmark vertices
        return torch.tensor(compute_geodesic_distances(mesh,source_index=np.array(args.landmarks))).T.to(device)    
