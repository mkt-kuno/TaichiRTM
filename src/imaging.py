# TaichiRTM
# Copyright (C) 2025 Yutaro Hara
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library. If not, see <https://www.gnu.org/licenses/>.

"""
Imaging utilities for RTM results.
Provides functions for loading, processing, and preparing RTM data for visualization.
"""

import glob
import os
from typing import Any, Dict, List, Optional

import numpy as np


def load_rtm_results(directory: str, pattern: str = '*.npz') -> List[Dict[str, Any]]:
    """
    Load RTM result files from a directory.
    
    Parameters
    ----------
    directory : str
        Directory containing npz files
    pattern : str
        Glob pattern for files (default: '*.npz')
        
    Returns
    -------
    list
        List of dictionaries containing RTM data
    """
    rtm_files = glob.glob(os.path.join(directory, pattern))
    results = []

    for filepath in rtm_files:
        data = np.load(filepath)
        result = {key: data[key] for key in data.files}
        result['filepath'] = filepath
        results.append(result)

    return results


def stack_rtm_images(results: List[Dict[str, Any]],
                     subtract_mean: bool = True) -> Dict[str, np.ndarray]:
    """
    Stack multiple RTM images.
    
    Parameters
    ----------
    results : list
        List of RTM result dictionaries
    subtract_mean : bool
        Whether to subtract mean from stacked images
        
    Returns
    -------
    dict
        Dictionary containing stacked u, v, w images and metadata
    """
    if not results:
        raise ValueError("No results to stack")

    u_sum = None
    v_sum = None
    w_sum = None

    for i, data in enumerate(results):
        u = data['u']
        v = data['v']
        w = data['w']

        if i == 0:
            u_sum = np.zeros_like(u)
            v_sum = np.zeros_like(v)
            w_sum = np.zeros_like(w)

        u_sum += u
        v_sum += v
        w_sum += w

    umap = u_sum.T
    vmap = v_sum.T
    wmap = w_sum.T

    if subtract_mean:
        umap = umap - np.mean(umap)
        vmap = vmap - np.mean(vmap)
        wmap = wmap - np.mean(wmap)

    last_data = results[-1]
    offset = float(last_data['offset'])
    dx = float(last_data['dx'])
    dz = float(last_data['dz'])
    nx = int(last_data['nx'])
    nz = int(last_data['nz'])

    xmin = -offset
    xmax = dx * nx - offset
    zmin = 0.0
    zmax = dz * nz

    return {
        'u': umap,
        'v': vmap,
        'w': wmap,
        'xmin': xmin,
        'xmax': xmax,
        'zmin': zmin,
        'zmax': zmax,
        'extent': [xmin, xmax, zmax, zmin],
        'dx': dx,
        'dz': dz,
        'nx': nx,
        'nz': nz
    }


def compute_display_limits(data: np.ndarray) -> tuple:
    """
    Compute symmetric display limits for seismic data.
    
    Parameters
    ----------
    data : np.ndarray
        Data array
        
    Returns
    -------
    tuple
        (vmin, vmax) for symmetric colormap
    """
    data_max = np.max(np.abs(data))
    return -data_max, data_max


def prepare_image_data(image: np.ndarray,
                       attenuate_region: Optional[tuple] = None,
                       attenuation_factor: float = 1e-3) -> np.ndarray:
    """
    Prepare image data for display.
    
    Parameters
    ----------
    image : np.ndarray
        RTM image
    attenuate_region : tuple, optional
        Region to attenuate (x_start, x_end, z_start, z_end)
    attenuation_factor : float
        Attenuation multiplier
        
    Returns
    -------
    np.ndarray
        Processed image
    """
    result = image.copy()

    if attenuate_region is not None:
        x_start, x_end, z_start, z_end = attenuate_region
        # Bounds checking for safe array slicing
        x_start = max(0, x_start)
        x_end = min(result.shape[0], x_end)
        z_start = max(0, z_start)
        z_end = min(result.shape[1], z_end)
        if x_start < x_end and z_start < z_end:
            result[x_start:x_end, z_start:z_end] *= attenuation_factor

    return result


def create_visualization_data(rtm_instance,
                              subtract_mean: bool = True,
                              attenuate_source: bool = True,
                              source_attenuation_radius: int = 10) -> Dict[str, Any]:
    """
    Create visualization data from RTM instance.
    
    Parameters
    ----------
    rtm_instance : ReverseTimeMigration
        RTM instance with results
    subtract_mean : bool
        Whether to subtract mean
    attenuate_source : bool
        Whether to attenuate near source location
    source_attenuation_radius : int
        Radius for source attenuation
        
    Returns
    -------
    dict
        Dictionary with visualization data
    """
    u = rtm_instance.image_u.copy()
    v = rtm_instance.image_v.copy()
    w = rtm_instance.image_w.copy()

    if attenuate_source and hasattr(rtm_instance, 'src_loc_step'):
        src = rtm_instance.src_loc_step[0]
        r = source_attenuation_radius

        x_start = max(0, src[0] - r)
        x_end = min(u.shape[0], src[0] + r)
        z_start = src[1]
        z_end = min(u.shape[1], src[1] + r)

        u[x_start:x_end, z_start:z_end] *= 1e-3
        v[:, z_start:z_end] *= 1e-1
        w[x_start:x_end, z_start:z_end] *= 1e-3

    umap = u.T
    vmap = v.T
    wmap = w.T

    if subtract_mean:
        umap = umap - np.mean(umap)
        vmap = vmap - np.mean(vmap)
        wmap = wmap - np.mean(wmap)

    extent = rtm_instance.get_axes_extent()

    return {
        'u': umap,
        'v': vmap,
        'w': wmap,
        'xmin': extent['xmin'],
        'xmax': extent['xmax'],
        'zmin': extent['zmin'],
        'zmax': extent['zmax'],
        'extent': [extent['xmin'], extent['xmax'], extent['zmax'], extent['zmin']],
        'u_lim': compute_display_limits(umap),
        'v_lim': compute_display_limits(vmap),
        'w_lim': compute_display_limits(wmap)
    }


def save_stacked_results(data: Dict[str, Any],
                         filepath: str):
    """
    Save stacked RTM results.
    
    Parameters
    ----------
    data : dict
        Stacked image data
    filepath : str
        Output filepath
    """
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    np.savez_compressed(
        filepath,
        u=data['u'],
        v=data['v'],
        w=data['w'],
        xmin=data['xmin'],
        xmax=data['xmax'],
        zmin=data['zmin'],
        zmax=data['zmax']
    )
    print(f'Stacked results saved to {filepath}')
