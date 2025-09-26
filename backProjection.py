#!/usr/bin/env python3
"""
nlos_backprojection.py

Implementation of the backprojection reconstruction algorithm for
Non-Line-Of-Sight (NLOS) 3D shape recovery, following Velten et al.,
"Recovering three-dimensional shape around a corner using ultrafast
time-of-flight imaging" (Nature Communications, 2012). See the
paper for detailed description of experimental setup and modeling. :contentReference[oaicite:1]{index=1}

Features implemented:
- Geometry and voxel-grid setup (oriented bounding box)
- Loading streak images + time axis
- Jitter/time-shift correction using calibration spot (simple peak alignment)
- Backprojection: for each voxel, sum weighted intensities from pixels
  whose time-of-flight matches voxel path length (with attenuation compensation)
- Filter: second derivative along depth (z)
- Confidence map computation (local normalization + soft threshold)
- Simple visualizations: depth map, confidence point cloud, and volume slices
- Small synthetic demo mode (if you don't have raw data yet)

How to use (examples):
  - python nlos_backprojection.py --demo
  - python nlos_backprojection.py --data_dir ./my_data --out results.npz

Data expected if not using --demo:
  - data_dir/streak_000.npy, streak_001.npy, ... (2D arrays: spatial x time)
    * or png/tiff images (will be converted to float)
  - data_dir/laser_positions.csv    (Nx,Ny,Nz columns or x,y,z rows per laser)
  - data_dir/pixel_wall_coords.csv  (mapping from camera spatial pixel -> 3D wall point)
  - data_dir/time_axis.npy          (1D array of time values corresponding to time bins)
  - camera center position and other geometry passed via args or file
"""

import os, sys
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
import imageio
import argparse

C = 299792458.0  # speed of light (m/s)

###########################
# Utilities / IO
###########################

def load_streak_images_from_dir(dirname):
    """
    Load streak images from a directory. Accepts .npy arrays, or .png/.tiff/.jpg images.
    Returns list of arrays (float32) and list of filenames sorted lexicographically.
    Each image is assumed shape (spatial_pixels, time_bins).
    """
    files = sorted([f for f in os.listdir(dirname) if f.lower().endswith(('.npy','.png','.jpg','.jpeg','.tif','.tiff'))])
    imgs = []
    for f in files:
        fp = os.path.join(dirname, f)
        if f.lower().endswith('.npy'):
            a = np.load(fp).astype(np.float32)
        else:
            a = imageio.imread(fp).astype(np.float32)
            # If color, convert to grayscale
            if a.ndim == 3:
                a = np.mean(a, axis=2)
        imgs.append(a)
    return imgs, files

def save_results(outpath, **kwargs):
    np.savez_compressed(outpath, **kwargs)
    print(f"Saved results to {outpath}")

def normalize_img(img):
    img = img - np.min(img)
    if np.max(img) > 0:
        img = img / np.max(img)
    return img

###########################
# Timing / calibration helpers
###########################

def detect_calibration_peak(streak_img, calib_spatial_idx=None, time_axis=None, threshold_frac=0.2):
    """
    Detect calibration/time-reference peak in a streak image.
    - streak_img: 2D (spatial, time)
    Returns time index (float) of detected calibration peak (centroid).
    If calib_spatial_idx is provided, will consider only that spatial index.
    """
    if calib_spatial_idx is not None:
        trace = streak_img[calib_spatial_idx, :]
    else:
        # collapse spatially to find bright spot
        trace = np.max(streak_img, axis=0)
    # basic thresholding + centroid
    tmax = np.max(trace)
    if tmax <= 0:
        return None
    thr = threshold_frac * tmax
    mask = trace >= thr
    if np.sum(mask) == 0:
        return np.argmax(trace)
    idxs = np.arange(len(trace))[mask]
    centroid = np.sum(idxs * trace[mask]) / np.sum(trace[mask])
    if time_axis is not None:
        return centroid if time_axis is None else np.interp(centroid, np.arange(len(time_axis)), time_axis)
    return centroid

def correct_jitter(streak_imgs, calib_spatial_idx, time_axis):
    """
    Align streak images using calibration spot (simple shift).
    - streak_imgs: list of 2D arrays (spatial x time)
    - calib_spatial_idx: int
    - time_axis: 1D array
    Returns shifted images (time-axis aligned by shifting columns).
    NOTE: shifts are integer-sample shifts here; could be improved with interpolation.
    """
    ref_time_idx = detect_calibration_peak(streak_imgs[0], calib_spatial_idx)
    out = []
    for im in streak_imgs:
        ti = detect_calibration_peak(im, calib_spatial_idx)
        if ti is None:
            shift = 0
        else:
            shift = int(round(ref_time_idx - ti))
        if shift == 0:
            out.append(im.copy())
        elif shift > 0:
            out.append(np.pad(im, ((0,0),(shift,0)), mode='constant')[:, :im.shape[1]])
        else:
            out.append(np.pad(im, ((0,0),(0,-shift)), mode='constant')[:, -shift:])
    return out

###########################
# Geometry / voxel grid
###########################

def make_voxel_grid(bounds_min, bounds_max, voxel_size):
    """
    Create a voxel grid (x,y,z) given bounds (in meters) and voxel size (m).
    Returns:
      xs, ys, zs: 1D arrays of coordinate axes
      grid_coords: (N_voxels, 3) array of voxel center coordinates
      shape: tuple (nx, ny, nz)
    """
    xs = np.arange(bounds_min[0], bounds_max[0]+1e-12, voxel_size)
    ys = np.arange(bounds_min[1], bounds_max[1]+1e-12, voxel_size)
    zs = np.arange(bounds_min[2], bounds_max[2]+1e-12, voxel_size)
    nx, ny, nz = len(xs), len(ys), len(zs)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='xy')
    coords = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T  # (N,3)
    return xs, ys, zs, coords, (nx, ny, nz)

###########################
# Backprojection core
###########################

def backproject_volume(streak_imgs, time_axis, laser_positions, pixel_wall_coords,
                       camera_center, voxel_coords, voxel_shape,
                       voxel_size, alpha=1.0, speed_of_light=C, verbose=True):
    """
    Core backprojection:
    - streak_imgs: list of 2D arrays, shape (spatial_pix_count, time_bins)
    - time_axis: 1D array of times for time bins (seconds)
    - laser_positions: list/array of 3D points for each streak image (meters)
    - pixel_wall_coords: array of shape (spatial_pix_count, 3) giving each spatial pixel's 3D position on the wall
    - camera_center: 3-vector of camera COP in meters
    - voxel_coords: (N_vox,3) coordinates of voxel centers
    - voxel_shape: (nx,ny,nz)
    - voxel_size: scalar (meters)
    - alpha: attenuation exponent (paper used alpha = 1 recommended)
    Returns:
      heatmap: 3D array same shape as voxel_shape with summed weighted votes
    """
    n_images = len(streak_imgs)
    n_vox = voxel_coords.shape[0]
    nx, ny, nz = voxel_shape
    heat = np.zeros(n_vox, dtype=np.float32)

    # Precompute |w - camera| distances for each spatial pixel
    pw = np.asarray(pixel_wall_coords)  # shape (spatial, 3)
    r4_arr = np.linalg.norm(pw - camera_center[np.newaxis,:], axis=1)  # (spatial,)

    # For faster indexing: flatten time axis
    time_bins = len(time_axis)
    dt = np.gradient(time_axis)  # not used but available

    # Build an acceleration scheme: for each image, iterate voxels and map TOF -> time index.
    # This is O(n_images * n_vox). For moderate sizes (e.g. 1e6 voxels) this is slow; for demo it's OK.
    # In practice implement spatial culling or GPU acceleration.
    for img_idx in tqdm(range(n_images), desc="Backproject images", disable=not verbose):
        img = streak_imgs[img_idx]  # shape (spatial, time)
        L = np.asarray(laser_positions[img_idx])  # laser spot on wall (3,)
        # Precompute distances from laser spot to voxels (r2) and voxels to wall points (r3 for each spatial pixel)
        # r2: (n_vox,)
        r2 = np.linalg.norm(voxel_coords - L[np.newaxis,:], axis=1)  # distance L -> s
        # For r3, we need distances from voxels to each wall pixel location; doing for all pixel x voxels might be large.
        # We'll loop over spatial pixels and accumulate contributions.
        spatial_count = img.shape[0]
        for px in range(spatial_count):
            w = pw[px]
            r3 = np.linalg.norm(voxel_coords - w[np.newaxis,:], axis=1)  # (n_vox,)
            total_dist = r2 + r3 + r4_arr[px]  # note: r1 (laser->wall) is constant per experiment and omitted here (will shift time axis if needed)
            tof = total_dist / speed_of_light  # seconds
            # Map tof to nearest time bin index
            # Use linear interpolation: get float bin index
            bin_idx = np.interp(tof, time_axis, np.arange(time_bins), left=-1, right=time_bins)
            # Now sample image intensity at (px, bin_idx) using linear interpolation along time axis
            # where bin_idx outside [0, time_bins-1] contribute zero
            valid_mask = (bin_idx >= 0) & (bin_idx < time_bins-1)
            if not np.any(valid_mask):
                continue
            # interpolate intensity
            bin_lo = np.floor(bin_idx[valid_mask]).astype(int)
            frac = bin_idx[valid_mask] - bin_lo
            intensities = (1 - frac) * img[px, bin_lo] + frac * img[px, bin_lo + 1]
            # attenuation compensation weight: (r2 * r3)**alpha
            weights = (r2[valid_mask] * r3[valid_mask]) ** alpha
            # accumulate weighted intensities into heat
            heat[valid_mask] += intensities * weights
    heatmap = heat.reshape(voxel_shape, order='C')  # shape (nx, ny, nz)
    return heatmap

###########################
# Filtering + confidence
###########################

def second_derivative_z(volume, axis=2):
    """
    Compute discrete second derivative along depth axis (z).
    Assumes volume shape (..., nz) and axis indicates axis index for z.
    Implementation via convolution [1, -2, 1] along that axis.
    """
    kernel = np.array([1.0, -2.0, 1.0], dtype=np.float32)
    return ndimage.convolve1d(volume, kernel, axis=axis, mode='reflect')

def compute_confidence(filtered_vol, local_window=(20,20,20), V0_frac=0.3):
    """
    Compute confidence map per paper:
      V' = tanh(20 * (V - V0)) * V / mloc
    where mloc is local maximum inside sliding window, V0 = V0_frac * global_max
    Returns confidence float array (same shape).
    """
    V = filtered_vol
    global_max = np.max(V)
    V0 = V0_frac * global_max
    # local maximum filter
    mloc = ndimage.maximum_filter(V, size=local_window, mode='reflect')
    # avoid division by zero
    denom = mloc.copy()
    denom[denom <= 1e-12] = 1e-12
    out = np.tanh(20.0 * (V - V0)) * V / denom
    return out

###########################
# Visualization
###########################

def depth_map_from_filtered(filtered_vol, zs):
    """
    For each (x,y), choose depth index of maximum filter response,
    and convert to depth coordinate (meters).
    Returns depth_map (nx, ny) and index_map (nx, ny)
    """
    idx = np.argmax(filtered_vol, axis=2)
    depth_map = zs[idx]
    return depth_map, idx

def plot_depth_map(depth_map, xs, ys, title="Depth map"):
    plt.figure(figsize=(6,6))
    plt.imshow(depth_map.T, origin='lower', extent=(xs[0], xs[-1], ys[0], ys[-1]), aspect='auto')
    plt.colorbar(label='depth (m)')
    plt.xlabel('x (m)'); plt.ylabel('y (m)')
    plt.title(title)
    plt.show()

def plot_confidence_pointcloud(confidence_vol, xs, ys, zs, threshold=0.1, max_points=200000):
    # sample voxels above threshold
    coords = np.array(np.where(confidence_vol > threshold)).T  # indices (i,j,k)
    if coords.size == 0:
        print("No voxels exceed confidence threshold.")
        return
    pts = np.column_stack((xs[coords[:,0]], ys[coords[:,1]], zs[coords[:,2]]))
    vals = confidence_vol[confidence_vol > threshold]
    # subsample if too many
    if pts.shape[0] > max_points:
        sel = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[sel]
        vals = vals[sel]
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=vals, cmap='viridis', s=1)
    fig.colorbar(sc, label='confidence')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
    plt.title('Confidence point cloud')
    plt.show()

###########################
# Example synthetic/demo data generator
###########################

def make_synthetic_demo(nx_spatial=80, nt=600, n_lasers=30, voxel_size=0.01):
    """
    Create a tiny synthetic demo: a small planar hidden object, a wall at z=0,
    camera at negative z looking at wall, laser scanning spots on wall.
    Returns streak_imgs (list), time_axis, laser_positions, pixel_wall_coords, camera_center.
    All distances in meters, time in seconds.
    """
    # geometry
    camera_center = np.array([0.0, -0.4, 0.62])  # example (x,y,z) in meters
    # wall plane z=0, wall spans x,y
    xs_wall = np.linspace(-0.2, 0.2, nx_spatial)
    wall_y = 0.0
    pixel_wall_coords = np.column_stack((xs_wall, np.full_like(xs_wall, wall_y), np.zeros_like(xs_wall)))
    # laser positions: along y = 0.1 line (slightly above camera line)
    laser_positions = [np.array([x, 0.1, 0.0]) for x in np.linspace(-0.2, 0.2, n_lasers)]
    # hidden object: small vertical square plane at z=0.25 in front of wall
    # We'll generate idealized time-of-flight signals for each laser & spatial pixel by summing contributions
    hidden_pts = []
    hx = np.linspace(-0.03, 0.03, 8)
    hy = np.linspace(-0.02, 0.02, 6)
    hz = 0.25
    for xx in hx:
        for yy in hy:
            hidden_pts.append(np.array([xx, yy, hz]))
    # time axis
    tmax = 3e-9
    time_axis = np.linspace(0.0, tmax, nt)
    streak_imgs = []
    for L in laser_positions:
        img = np.zeros((nx_spatial, nt), dtype=np.float32)
        for s in hidden_pts:
            r2 = np.linalg.norm(s - L)
            for px, w in enumerate(pixel_wall_coords):
                r3 = np.linalg.norm(s - w)
                r4 = np.linalg.norm(w - camera_center)
                total = r2 + r3 + r4
                tof = total / C
                # inject a small Gaussian pulse on that time-bin for that pixel
                tbin = np.interp(tof, time_axis, np.arange(nt))
                tgrid = np.arange(nt)
                pulse = np.exp(-0.5 * ((tgrid - tbin)/2.5)**2) * 5e-6  # amplitude small
                img[px,:] += pulse
        # add calibration spot at spatial idx 10 (direct path short)
        cal_px = nx_spatial//8
        r_cal = np.linalg.norm(np.array([0.0,0.0,0.62]) - np.array([0.0,0.0,0.0]))  # fake
        tcal = 0.9e-9
        img[cal_px,:] += np.exp(-0.5*((np.arange(nt)-np.interp(tcal, time_axis, np.arange(nt)))/2)**2)*1.0
        # add noise
        img += np.random.randn(*img.shape) * 1e-7
        streak_imgs.append(img)
    return streak_imgs, time_axis, laser_positions, pixel_wall_coords, camera_center

###########################
# Main / CLI
###########################

def parse_args():
    p = argparse.ArgumentParser(description="NLOS backprojection reconstruction (Velten et al., 2012).")
    p.add_argument('--data_dir', default=None, help='Directory with streak images and geometry files (see README in header).')
    p.add_argument('--out', default='nlos_results.npz', help='Output .npz file')
    p.add_argument('--voxel_size', type=float, default=0.008, help='Voxel size in meters (default 8 mm)')
    p.add_argument('--alpha', type=float, default=1.0, help='Attenuation exponent (paper used alpha=1)')
    p.add_argument('--demo', action='store_true', help='Run synthetic demo (no external data required)')
    p.add_argument('--verbose', action='store_true', help='Verbose progress info')
    return p.parse_args()

def main():
    args = parse_args()
    if args.demo:
        print("Running synthetic demo...")
        streak_imgs, time_axis, laser_positions, pixel_wall_coords, camera_center = make_synthetic_demo()
        calib_spatial_idx = streak_imgs[0].shape[0]//8
        # bounding box (x,y,z) roughly in front of wall from x=-0.1..0.1, y=-0.1..0.1, z=0.05..0.45
        bounds_min = (-0.15, -0.12, 0.05)
        bounds_max = ( 0.15,  0.12, 0.45)
        voxel_size = args.voxel_size
    else:
        if args.data_dir is None:
            print("Error: --data_dir required unless --demo. See header docstring.")
            return
        # load streak images
        streak_imgs, files = load_streak_images_from_dir(args.data_dir)
        # try to load geometry files
        try:
            laser_positions = np.loadtxt(os.path.join(args.data_dir, 'laser_positions.csv'), delimiter=',')
            pixel_wall_coords = np.loadtxt(os.path.join(args.data_dir, 'pixel_wall_coords.csv'), delimiter=',')
            time_axis = np.load(os.path.join(args.data_dir, 'time_axis.npy'))
            camera_center = np.loadtxt(os.path.join(args.data_dir, 'camera_center.csv'), delimiter=',')
            calib_spatial_idx = 0  # user can change this in code; could be read from file
            voxel_size = args.voxel_size
            # bounds: either in file or set by user; we set a default and user can adapt
            bounds_min = (-0.2, -0.2, 0.05)
            bounds_max = ( 0.2,  0.2, 0.45)
        except Exception as e:
            raise RuntimeError("Failed to load geometry/time files from data_dir. See README and provide required files.") from e

    # Preprocess: normalize and jitter-correct
    streak_imgs = [normalize_img(im) for im in streak_imgs]
    print("Applying jitter/time alignment (calibration spot)...")
    streak_imgs = correct_jitter(streak_imgs, calib_spatial_idx, time_axis)

    # Create voxel grid (coarse -> optional refine)
    print("Creating voxel grid (bounds_min={}, bounds_max={}, voxel_size={} m)".format(bounds_min, bounds_max, voxel_size))
    xs, ys, zs, coords, shape = make_voxel_grid(bounds_min, bounds_max, voxel_size)
    print("Voxel grid shape:", shape, "Total voxels:", coords.shape[0])

    # Backprojection
    print("Starting backprojection (this may take time for large voxel grids).")
    heatmap = backproject_volume(streak_imgs, time_axis, laser_positions, pixel_wall_coords, camera_center, coords, shape, voxel_size, alpha=args.alpha, verbose=args.verbose)

    # Filtering: second derivative along z (axis index 2 for shape (nx,ny,nz))
    print("Applying 2nd derivative filter along depth (z)...")
    filtered = second_derivative_z(heatmap, axis=2)
    # Normalize filter response for stability
    filtered = filtered / (np.max(np.abs(filtered)) + 1e-12)

    # Confidence map
    print("Computing confidence map...")
    conf = compute_confidence(filtered, local_window=(20,20,20), V0_frac=0.3)

    # Depth map
    depth_map, idx_map = depth_map_from_filtered(filtered, zs)

    # Save results
    save_results(args.out, heatmap=heatmap, filtered=filtered, confidence=conf, depth_map=depth_map, xs=xs, ys=ys, zs=zs)

    # Visualization
    print("Showing depth map...")
    plot_depth_map(depth_map, xs, ys, title="Depth map (from filtered heatmap)")
    print("Showing confidence point cloud (threshold=0.2)...")
    plot_confidence_pointcloud(conf, xs, ys, zs, threshold=0.2)

if __name__ == '__main__':
    main()
