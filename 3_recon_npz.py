#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 13:18:09 2025

@author: alex
"""

"""
Quick 3D reconstruction preview from ute3d_radial.npz

Fast path: FINUFFT (if installed)  -> pip install finufft
Fallback : Kaiser–Bessel gridding  -> pure NumPy (slower)
Saves: recon_mag.png
"""

import numpy as np
import matplotlib.pyplot as plt

npz_path = "ute3d_radial.npz"  # change if needed
ZSLICE = None  # e.g., set to an int to show a specific axial slice

# -------------------- Load data -------------------- #
dat = np.load(npz_path, allow_pickle=True)
kdata = dat["kdata"]              # (M,) complex64
ktraj = dat["ktraj"]              # (M,3) float32 in cycles/FOV ~ [-kmax,kmax]
dcf   = dat["dcf"].astype(np.float64)      # (M,)
meta  = dat["meta"].item()

N       = int(meta.get("N", 160))
osf     = float(meta.get("os_forward", 2.0))
kmax    = float(meta.get("kmax", 0.5))
units   = meta.get("ktraj_units", "cycles/FOV")

# Center-out vs symmetric doesn’t matter for adjoint preview, but kmax is needed
# FINUFFT expects coordinates in radians in [-pi, pi]; our ktraj is in cycles/FOV
# Convert cycles/FOV -> radians: 2*pi * k (cycles) mapped to [-2*pi*kmax, 2*pi*kmax]
kx = (2*np.pi) * ktraj[:,0]
ky = (2*np.pi) * ktraj[:,1]
kz = (2*np.pi) * ktraj[:,2]

# Weight by DCF for a reasonable backprojection
samps = (kdata.astype(np.complex128) * dcf).ravel()

# -------------------- Try FINUFFT path -------------------- #
recon = None
try:
    import finufft  # pip install finufft
    # Type-1 NUFFT: nonuniform points -> uniform grid (Fourier coefficients)
    # Output grid size (oversampled), then crop
    Nx = Ny = Nz = int(np.round(osf * N))

    # FINUFFT uses frequency grid indices centered at 0, size (2J+1) style;
    # We’ll request standard (Nx,Ny,Nz) coeffs and then crop to N^3
    print("[info] Using FINUFFT for fast adjoint NUFFT...")
    c = finufft.nufft3d1(kx, ky, kz, samps, (Nx, Ny, Nz), eps=1e-6)  # complex grid

    # Shift to image domain by inverse FFT (adjoint ≈ backprojection)
    img_os = np.fft.ifftn(np.fft.ifftshift(c))
    # Crop center to N^3
    cx0, cy0, cz0 = Nx//2, Ny//2, Nz//2
    half = N//2
    img = img_os[cx0-half:cx0-half+N, cy0-half:cy0-half+N, cz0-half:cz0-half+N]

    recon = img

except Exception as e:
    print(f"[warn] FINUFFT unavailable or failed ({e}). Falling back to Kaiser–Bessel gridding (slow)...")

# -------------------- Fallback: KB gridding -------------------- #
if recon is None:
    # Very simple separable Kaiser–Bessel (KB) gridding.
    # NOTE: This is a *slow* Python implementation for preview/debugging only.
    Nx = Ny = Nz = int(np.round(osf * N))
    grid = np.zeros((Nx, Ny, Nz), dtype=np.complex128)

    # Map k-space from cycles/FOV [-kmax,kmax] -> grid index [0,Nx)
    # cycles/FOV scaled by osf*N to grid; shift by Nx/2 to center
    # Our ktraj is in cycles/FOV already.
    gx = (ktraj[:,0] * (osf * N) + Nx/2.0)
    gy = (ktraj[:,1] * (osf * N) + Ny/2.0)
    gz = (ktraj[:,2] * (osf * N) + Nz/2.0)

    # KB kernel parameters
    W = 4.0         # kernel width (grid units)
    halfW = int(np.ceil(W/2))
    beta = 13.855   # typical for W=4, osf≈2 (Jackson 1991 style)
    # 1D Kaiser–Bessel kernel
    def kb(u):
        # u is distance in grid units; support |u| <= W/2
        x = np.sqrt((1 - (2*u/W)**2).clip(0,1))
        return np.i0(beta * x)  # unnormalized; fine for preview
    # Gridding loop (vectorized per small neighborhood)
    # WARNING: O(M * W^3) — fine for a few hundred thousand samples, but not “fast”.
    M = samps.shape[0]
    for m in range(M):
        x0 = int(np.floor(gx[m]))
        y0 = int(np.floor(gy[m]))
        z0 = int(np.floor(gz[m]))
        xs = range(x0 - halfW + 1, x0 + halfW + 1)
        ys = range(y0 - halfW + 1, y0 + halfW + 1)
        zs = range(z0 - halfW + 1, z0 + halfW + 1)
        for xi in xs:
            dx = abs(gx[m] - xi)
            if dx > W/2: 
                continue
            wx = kb(dx)
            xw = xi % Nx
            for yi in ys:
                dy = abs(gy[m] - yi)
                if dy > W/2:
                    continue
                wy = kb(dy)
                yw = yi % Ny
                wxy = wx * wy
                for zi in zs:
                    dz = abs(gz[m] - zi)
                    if dz > W/2:
                        continue
                    zw = zi % Nz
                    wz = kb(dz)
                    grid[xw, yw, zw] += samps[m] * (wxy * wz)

    # Simple roll-off correction (separable deapodization) on the grid domain
    # Build 1D deapodization along each axis (approximate)
    def kb_deapo_vec(n):
        # frequency indices centered at 0
        u = np.arange(n) - n/2.0
        u = np.fft.fftshift(u)
        # normalize to grid units (one grid step = 1.0)
        # approximate inverse of kb at those locations
        v = np.abs(u)  # distance
        w = np.ones_like(v, dtype=np.float64)
        mask = v <= W/2
        vv = v[mask]
        x = np.sqrt((1 - (2*vv/W)**2).clip(0,1))
        w[mask] = np.i0(beta * x)
        w[~mask] = 1e6  # avoid amplifying outside support (won’t be used)
        w[w==0] = 1e-6
        return 1.0 / w

    dx = kb_deapo_vec(Nx)
    dy = kb_deapo_vec(Ny)
    dz = kb_deapo_vec(Nz)
    # Apply separable deapodization
    grid *= dx[:,None,None]
    grid *= dy[None,:,None]
    grid *= dz[None,None,:]

    # Back to image by IFFT and crop
    img_os = np.fft.ifftn(np.fft.ifftshift(grid))
    cx0, cy0, cz0 = Nx//2, Ny//2, Nz//2
    half = N//2
    recon = img_os[cx0-half:cx0-half+N, cy0-half:cy0-half+N, cz0-half:cz0-half+N]

# -------------------- Display & Save -------------------- #
mag = np.abs(recon)
mag /= (mag.max() + 1e-12)

# choose a middle slice if not specified
if ZSLICE is None:
    ZSLICE = N // 2

plt.figure(figsize=(6,6))
plt.imshow(mag[:,:,ZSLICE].T, origin="lower", cmap="gray")
plt.title(f"Adjoint NUFFT Preview (slice {ZSLICE})")
plt.axis("off")
plt.tight_layout()
plt.savefig("recon_mag.png", dpi=300)
plt.show()

print("Saved: recon_mag.png")


import nibabel as nib
import numpy as np

# assume recon (complex image) is from NUFFT/gridding
mag = np.abs(recon)
mag = mag / (mag.max() + 1e-12)  # normalize [0,1] for preview

# --- Define voxel size ---
# If FOV = N (matrix size) / kmax*2 (cycles/FOV),
# you can compute approximate voxel spacing.
N = mag.shape[0]
kmax = float(meta.get("kmax", 0.5))  # cycles/FOV
fov = N / (2*kmax)   # in arbitrary units (usually mm if ktraj scaled properly)
voxel_size = fov / N
print(f"Voxel size ~ {voxel_size:.3f} (same units as FOV)")

# --- Create NIfTI image ---
affine = np.diag([voxel_size, voxel_size, voxel_size, 1.0])  # simple isotropic affine
nii = nib.Nifti1Image(mag.astype(np.float32), affine)

# --- Save ---
nib.save(nii, "recon_mag.nii.gz")
print("Saved: recon_mag.nii.gz")

