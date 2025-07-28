import os
import torch
import numpy as np
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from plotly.subplots import make_subplots



def create_rgb_colormap(points):
    """
    Create a colormap for the points based on their coordinates.
    The color is determined by the distance from the origin.
    """
    points = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))
    colors_hex = [mcolors.rgb2hex(c) for c in points]
    
    return colors_hex 


def plot_target(filename, points, plots_path, show=False):
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color='blue'
        )
    )])
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title=filename,
        width=800,
        height=800
    )
    if show: fig.show()
    
    fig.write_html(f'{plots_path}/{filename}.html')
    fig.write_image(f'{plots_path}/{filename}.png')


def start_end_subplot(x_0, x_1_estimated, run_name='Title', plots_path='./', show=False, html=False, png=False):
    x_0 = x_0.cpu().numpy()
    x_1_estimated = x_1_estimated.cpu().numpy()
    
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]], subplot_titles=("Source (x_0)", "Estimated (x_T)"))
    colors_hex = create_rgb_colormap(x_0)

    # Plot the initial point cloud
    fig.add_trace(go.Scatter3d(
        x=x_0[:, 0],
        y=x_0[:, 1],
        z=x_0[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=colors_hex,  # Corresponding colors for each point
        ),
        name='Initial Points'
    ), row=1, col=1)

    # Plot the transformed point cloud (from flow[-1])
    fig.add_trace(go.Scatter3d(
        x=x_1_estimated[:, 0],
        y=x_1_estimated[:, 1],
        z=x_1_estimated[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=colors_hex,  # Same colors to encode correspondence
        ),
        name='Transformed Points'
    ), row=1, col=2)


    # Adjust layout parameters to make the figure bigger
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        scene2=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title=run_name,
        width=1800,
        height=800
    )
    
    if show: fig.show()
       
    if html: fig.write_html(f'{plots_path}/{run_name}.html')
    if png: fig.write_image(f'{plots_path}/{run_name}.png')
    
    
def plot_points(points, run_name, plots_path, title, show=False):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    
    colors_hex = create_rgb_colormap(points)
    fig = go.Figure()

    # Plot the point cloud
    fig.add_trace(go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=colors_hex,  # corresponding colors for each point
        ),
        name='Points'
    ))

    # Adjust layout parameters
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title=f"{run_name} - {title}",
        width=900,  # Half the width since we're only showing one plot
        height=800
    )
    
    if show: 
        fig.show()
       
    fig.write_image(f'{plots_path}/{run_name}.png')

   
def start_end_subplot_volume(x_0, x_1_estimated, run_name, plots_path, show=False):
    x_0 = x_0.cpu().numpy()
    x_1_estimated = x_1_estimated.cpu().numpy()
    
    # Compute distances from the center for x_0
    distances = ((x_0 ** 2).sum(axis=1)) ** 0.5
    norm = mcolors.Normalize(vmin=distances.min(), vmax=distances.max())
    colormap = cm.get_cmap('viridis')
    colors_hex = [mcolors.rgb2hex(colormap(norm(d))) for d in distances]

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]], subplot_titles=("Source (x_0)", "Estimated (x_T)"))

    # Plot the initial point cloud
    fig.add_trace(go.Scatter3d(
        x=x_0[:, 0],
        y=x_0[:, 1],
        z=x_0[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=colors_hex,  # Corresponding colors for each point
        ),
        name='Initial Points'
    ), row=1, col=1)

    # Plot the transformed point cloud (from flow[-1])
    fig.add_trace(go.Scatter3d(
        x=x_1_estimated[:, 0],
        y=x_1_estimated[:, 1],
        z=x_1_estimated[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=colors_hex,  # Same colors to encode correspondence
        ),
        name='Transformed Points'
    ), row=1, col=2)

    # Adjust layout parameters to make the figure bigger
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        scene2=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title=run_name,
        width=1800,
        height=800
    )
    
    if show: 
        fig.show()
       
    fig.write_html(f'{plots_path}/{run_name}_radius.html')
    fig.write_image(f'{plots_path}/{run_name}_radius.png')
    
def create_colormap(VERT):
    """
    Creates a uniform color map on a mesh

    Args:
        VERT (Nx3 ndarray): The vertices of the object to plot

    Returns:
        Nx3: The RGB colors per point on the mesh
    """
    VERT = np.double(VERT)
    minx = np.min(VERT[:, 0])
    miny = np.min(VERT[:, 1])
    minz = np.min(VERT[:, 2])
    maxx = np.max(VERT[:, 0])
    maxy = np.max(VERT[:, 1])
    maxz = np.max(VERT[:, 2])
    colors = np.stack(
        [
            ((VERT[:, 0] - minx) / (maxx - minx)),
            ((VERT[:, 1] - miny) / (maxy - miny)),
            ((VERT[:, 2] - minz) / (maxz - minz)),
        ]
    ).transpose()
    return colors

def to_rgb_strings(v):
    """
    Converts an (N,3) RGB array in [0,1] to Plotly-compatible 'rgb(r,g,b)' strings
    """
    v = np.clip(v, 0, 1)  # ensure in [0,1]
    v8 = (v * 255).astype(np.uint8)
    return [f"rgb({r},{g},{b})" for r, g, b in v8]


    
def source_target_plot(source, source_v, target, target_v,
                              run_name='Source vs Target RGB',
                              plots_path='./', show=True):
    """
    Plots source and target point clouds side by side, coloring points by their RGB values.
    Args:
        source: (N, 3) numpy array or tensor of source points
        source_v: (N, 3) numpy array or tensor of RGB values for source, values in [0,1]
        target: (M, 3) numpy array or tensor of target points
        target_v: (M, 3) numpy array or tensor of RGB values for target, values in [0,1]
        run_name: name for saving the plot
        plots_path: directory to save the plot
        show: whether to display the plot
    """

    # Convert to numpy if needed
    if hasattr(source, 'cpu'): source = source.cpu().numpy()
    if hasattr(source_v, 'cpu'): source_v = source_v.cpu().numpy()
    if hasattr(target, 'cpu'): target = target.cpu().numpy()
    if hasattr(target_v, 'cpu'): target_v = target_v.cpu().numpy()

    # Prepare RGB color lists
    #def to_rgb_strings(v):
        # v is (N,3), values in [0,1]
    #    v = np.clip(v, 0, 1)
        # scale to 0-255 ints
    #    v8 = (v * 255).astype(np.int32)
    #    return [f"rgb({r},{g},{b})" for r, g, b in v8]

    source_colors = to_rgb_strings(source_v) if source_v.ndim == 2 and source_v.shape[1] == 3 else None
    target_colors = to_rgb_strings(target_v) if target_v.ndim == 2 and target_v.shape[1] == 3 else None

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=("Source", "Target")
    )

    # Add traces
    fig.add_trace(
        go.Scatter3d(
            x=source[:, 0], y=source[:, 1], z=source[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=source_colors if source_colors is not None else source_v,
                colorscale=None if source_colors is not None else 'Plasma',
                showscale=False if source_colors is not None else True,
            ),
            name='Source'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter3d(
            x=target[:, 0], y=target[:, 1], z=target[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=target_colors if target_colors is not None else target_v,
                colorscale=None if target_colors is not None else 'Plasma',
                showscale=False if target_colors is not None else True,
            ),
            name='Target'
        ),
        row=1, col=2
    )

    # Layout
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        scene2=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        title=run_name,
        width=1800,
        height=800
    )

    # Show and save
    if show:
        fig.show()
    fig.write_html(f'{plots_path}/{run_name}_rgb.html')
    fig.write_image(f'{plots_path}/{run_name}_rgb.png')
