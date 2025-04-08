"""Functions for KBMOD's core search (shifts and stacking of images)."""

import numpy as np
import torch

from kbmod.filters.clustering_grid import TrajectoryClusterGrid
from kbmod.core.psf import convolve_psf_and_image
from kbmod.core.trajectory import Trajectory


def generate_psi_phi_images(im_stack, psfs=None):
    """Generate the PSI and PHI images from an image stack.

    Parameters
    ----------
    im_stack : `ImageStackPy`
        The image data to use.
    psfs : `list`, optional
        The length T array of PSF information. If None, uses
        the PSFs stored in im_stack.

    Returns
    -------
    psi : `numpy.ndarray`
        The PSI image.
    phi : `numpy.ndarray`
        The PHI image.
    """
    if psfs is not None and len(psfs) != im_stack.num_times:
        raise ValueError(f"PSF data must have {im_stack.num_times} entries.")

    psi = np.zeros_like(im_stack.sci)
    phi = np.zeros_like(im_stack.sci)
    valid_mask = ~(np.isnan(im_stack.sci) | np.isnan(im_stack.var) | (im_stack.var <= 0.0))

    psi[valid_mask] = im_stack.sci[valid_mask] / im_stack.var[valid_mask]
    phi[valid_mask] = 1.0 / im_stack.var[valid_mask]

    # Convolve each timestep with the corresponding PSF.
    for t in range(im_stack.num_times):
        psf = psfs[t] if psfs is not None else im_stack.psfs[t]
        psi[t, :, :] = convolve_psf_and_image(psi[t, :, :], psf)
        phi[t, :, :] = convolve_psf_and_image(phi[t, :, :], psf)

    # Mask the bad pixels.
    psi[~valid_mask] = np.nan
    phi[~valid_mask] = np.nan

    return psi, phi


def evaluate_trajectory_likelihood(x0, y0, dx, dy, inds, psi, phi):
    """Evaluate the likelihood of a trajectory given PSI and PHI images.

    Parameters
    ----------
    x0 : `int`
        The initial x-coordinate (in pixels).
    y0 : `int`
        The initial y-coordinate (in pixels).
    dx : `torch.tensor`
        A length T array of the x-offsets from the starting position
        at each time step (in pixels).
    dy : `torch.tensor`
        A length T array of the y-offsets from the starting position
        at each time step (in pixels).
    inds : `torch.tensor`
        An array of indices [0, T-1] for book keeping.
    psi : `torch.tensor`
        A T x H x W array of PSI values where T is the number of time steps,
        H is the image height, and W is the image width.
    phi : `torch.tensor`
        A T x H x W array of PHI values where T is the number of time steps,
        H is the image height, and W is the image width.

    Returns
    -------
    likelihood : `float`
        The likelihood of the trajectory starting at this pixel.
    """
    # Predict the positions from this pixel.
    xp = x0 + dx
    yp = y0 + dy

    # Flag the points that will be out of bound and replace the predictions with (0, 0)
    # in_bnds = (xp >= 0) & (xp < psi.shape[2]) & (yp >= 0) & (yp < psi.shape[1])
    xp = torch.where((xp >= 0) & (xp < psi.shape[2]), xp, psi.shape[2] - 1)
    yp = torch.where((yp >= 0) & (yp < psi.shape[1]), yp, psi.shape[1] - 1)

    # Get the psi and phi values at the trajectory's positions.
    psi_vals = psi[inds, yp, xp]
    phi_vals = phi[inds, yp, xp]
    likelihood = torch.nansum(psi_vals) / (torch.nansum(phi_vals) + 1e-28)
    count = torch.sum(~torch.isnan(psi_vals) & ~torch.isnan(phi_vals))

    return likelihood, count


def search_trajectories(
    stack,
    trj_list,
    min_obs=10,
    min_lh=0.001,
    cluster_eps=1.0,
    x_start=None,
    x_end=None,
    y_start=None,
    y_end=None,
    device=None,
):
    """Evaluate the likelihood of a trajectory given PSI and PHI images.

    Parameters
    ----------
    stack : `ImageStackPy`
        The image data to use.
    trj_list : `list` or `Trajectory`
        A list of Trajectory objects.
    min_obs : `int`, optional
        The minimum number of observations required to consider a trajectory.
        Default is 10.
    min_lh : `float`, optional
        The minimum likelihood required to consider a trajectory.
        Default is 0.001.
    cluster_eps : `float`, optional
        The maximum distance between two trajectories to be considered in the same cluster.
        Default is 1.0.
    x_start : `int`, optional
        The starting x-coordinate for the search (in pixels). If None 0 is used.
        Default is None.
    x_end : `int`, optional
        The ending x-coordinate for the search (in pixels). If None the width of the image is used.
        Default is None.
    y_start : `int`, optional
        The starting y-coordinate for the search (in pixels). If None 0 is used.
        Default is None.
    y_end : `int`, optional
        The ending y-coordinate for the search (in pixels). If None the height of the image is used.
        Default is None.
    device : `str`, optional
        The device to use for computation.
        Default is "cuda" if available, otherwise "cpu".

    Returns
    -------
    results : `list`
        A list of the best Trajectory objects.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Generate the psi and phi images. We pad the images with a row and col
    # of NaNs before turning them into tensors.
    psi, phi = generate_psi_phi_images(stack)
    psi = np.pad(psi, ((0, 0), (0, 1), (0, 1)), constant_values=np.nan)
    phi = np.pad(phi, ((0, 0), (0, 1), (0, 1)), constant_values=np.nan)
    psi = torch.tensor(psi, dtype=torch.float32, device=device)
    phi = torch.tensor(phi, dtype=torch.float32, device=device)

    # Create the vmapped function for evaluating the likelihood.
    lh_fun = torch.vmap(
        evaluate_trajectory_likelihood,
        in_dims=(0, 0, None, None, None, None, None),
        chunk_size=100_000,
    )
    inds = torch.tensor(range(stack.num_times), device=device)

    # Generate the list of pixel pairs to search.
    if x_start is None:
        x_start = 0
    if x_end is None:
        x_end = stack.width
    if y_start is None:
        y_start = 0
    if y_end is None:
        y_end = stack.height
    x_inds, y_inds = np.meshgrid(range(x_start, x_end), range(y_start, y_end), indexing="xy")
    x_inds = torch.tensor(x_inds.flatten(), device=device)
    y_inds = torch.tensor(y_inds.flatten(), device=device)

    # Evaluate the likelihood for each candidate velocity.
    times = stack.zeroed_times
    max_dt = np.max(times) - np.min(times)
    results_grid = TrajectoryClusterGrid(cluster_eps, max_dt)
    for trj in trj_list:
        # Predict the trajectory's offset at each time, centering in the pixel
        # and converting to an int.
        dx = torch.tensor(np.array(trj.vx * times + 0.5).astype(int), dtype=torch.int, device=device)
        dy = torch.tensor(np.array(trj.vy * times + 0.5).astype(int), dtype=torch.int, device=device)

        # Evaluate the likelihood of the trajectory.
        lh, counts = lh_fun(x_inds, y_inds, dx, dy, inds, psi, phi)

        # Get the trajectories that are above the two thresholds.
        mask = (lh > min_lh) & (counts >= min_obs)
        valid_lh = lh[mask].cpu()
        valid_obs = counts[mask].cpu()
        valid_x = x_inds[mask].cpu()
        valid_y = y_inds[mask].cpu()

        for x, y, lh, obs in zip(valid_x, valid_y, valid_lh, valid_obs):
            current = Trajectory(x=x, y=y, vx=trj.vx, vy=trj.vy, lh=lh, obs_count=obs)
            results_grid.add_trajectory(current)

    # Return a list of the best trajectories from each grid square.
    return results_grid.get_trajectories()


def search_trajectories_roll(
    stack,
    trj_list,
    min_obs=10,
    min_lh=0.001,
    cluster_eps=1.0,
    device=None,
):
    """Evaluate the likelihood of a trajectory given PSI and PHI images.

    Parameters
    ----------
    stack : `ImageStackPy`
        The image data to use.
    trj_list : `list` or `Trajectory`
        A list of Trajectory objects.
    min_obs : `int`, optional
        The minimum number of observations required to consider a trajectory.
        Default is 10.
    min_lh : `float`, optional
        The minimum likelihood required to consider a trajectory.
        Default is 0.001.
    cluster_eps : `float`, optional
        The maximum distance between two trajectories to be considered in the same cluster.
        Default is 1.0.
    device : `str`, optional
        The device to use for computation.
        Default is "cuda" if available, otherwise "cpu".

    Returns
    -------
    results : `list`
        A list of the best Trajectory objects.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # Get the stats
    img_w = stack.width
    img_h = stack.height

    # Generate the psi and phi images and make them into tensors.
    psi, phi = generate_psi_phi_images(stack)
    psi = torch.tensor(psi, dtype=torch.float32, device=device)
    phi = torch.tensor(phi, dtype=torch.float32, device=device)

    # Generate the list of pixel pairs to search.]
    x_inds = torch.tensor(range(img_w), device=device)
    y_inds = torch.tensor(range(img_h), device=device)
    grid_x, grid_y = torch.meshgrid(x_inds, y_inds, indexing='xy')

    # Evaluate the likelihood for each candidate velocity.
    times = stack.zeroed_times
    max_dt = np.max(times) - np.min(times)
    results_grid = TrajectoryClusterGrid(cluster_eps, max_dt)
    for trj in trj_list:
        phi_sum = torch.zeros((img_h, img_w), device=device)
        psi_sum = torch.full((img_h, img_w), 1e-28, device=device)
        counts = torch.zeros((img_h, img_w), device=device)

        # Shift each time by the correct amount.
        for t in range(len(times)):
            dx = int(trj.vx * times[t] + 0.5)
            dy = int(trj.vy * times[t] + 0.5)

            psi_roll = torch.roll(psi[t, :, :], shifts=(-dy, -dx), dims=(0, 1))
            phi_roll = torch.roll(phi[t, :, :], shifts=(-dy, -dx), dims=(0, 1))
            valid = torch.where(
                (
                    ~torch.isnan(psi_roll) &
                    ~torch.isnan(phi_roll) &
                    (phi_roll > 0.0) &
                    (grid_x + dx >= 0) & 
                    (grid_x + dx < img_w) & 
                    (grid_y + dy >= 0) & 
                    (grid_y + dy < img_h)
                ),
                1.0,
                0.0,
            )
            psi_sum += psi_roll * valid
            phi_sum += phi_roll * valid
            counts += valid
        
        # Compute the likelihood.
        lh = psi_sum / torch.sqrt(phi_sum)

        # Get the trajectories that are above the two thresholds.
        mask = (lh > min_lh) & (counts >= min_obs)
        valid_lh = lh[mask].cpu()
        valid_obs = counts[mask].cpu()
        valid_x = grid_x[mask].cpu()
        valid_y = grid_y[mask].cpu()

        for x, y, lh, obs in zip(valid_x, valid_y, valid_lh, valid_obs):
            current = Trajectory(x=x, y=y, vx=trj.vx, vy=trj.vy, lh=lh, obs_count=obs)
            results_grid.add_trajectory(current)

    # Return a list of the best trajectories from each grid square.
    return results_grid.get_trajectories()
