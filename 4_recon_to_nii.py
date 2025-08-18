#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 13:25:01 2025

@author: alex
"""

#!/usr/bin/env python3
"""
recon_to_nii.py
- Load k-space NPZ (kdata, ktraj, dcf, meta)
- Adjoint NUFFT backprojection (FINUFFT if available, else KB gridding)
- Save magnitude image as 3D NIfTI (.nii.gz) and a PNG preview.

Usage:
  python recon_to_nii.py --npz ute3d_radial.npz --fov-mm 20 --outfile recon_mag.nii.gz
"""

import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def try_finufft(kx, ky, kz, samps, N, osf):
    try:
        import finufft
        Nx = Ny = Nz = int(np.round(osf * N))
        print("[info] Using FINUFFT...")
        c = finufft.nufft3d1(kx, ky, kz, samps, (Nx, Ny, Nz), eps=1e-6)
        img_os = np.fft.ifftn(np.fft.ifftshift(c))
        cx, cy, cz = Nx//2, Ny//2, Nz//2
        half = N//2
        img = img_os[cx-half:cx-half+N, cy-half:cy-half+N, cz-half:cz-half+N]
        return img
    except Exception as e:
        print(f"[warn] FINUFFT unavailable/failed: {e}")
        return None

def kb_grid_fallback(ktraj, samps, N, osf, W=4.0, beta=13.855):
    # Very simple Kaiser–Bessel gridding (slow, preview quality)
    Nx = Ny = Nz = int(np.round(osf * N))
    grid = np.zeros((Nx, Ny, Nz), dtype=np.complex128)

    gx = (ktraj[:,0] * (osf * N) + Nx/2.0)
    gy = (ktraj[:,1] * (osf * N) + Ny/2.0)
    gz = (ktraj[:,2] * (osf * N) + Nz/2.0)

    halfW = int(np.ceil(W/2))
    def kb(u):
        x = np.sqrt((1 - (2*u/W)**2).clip(0,1))
        return np.i0(beta * x)

    M = samps.shape[0]
    for m in range(M):
        x0 = int(np.floor(gx[m])); y0 = int(np.floor(gy[m])); z0 = int(np.floor(gz[m]))
        for xi in range(x0-halfW+1, x0+halfW+1):
            dx = abs(gx[m] - xi); 
            if dx > W/2: continue
            wx = kb(dx); xw = xi % Nx
            for yi in range(y0-halfW+1, y0+halfW+1):
                dy = abs(gy[m] - yi)
                if dy > W/2: continue
                wy = kb(dy); yw = yi % Ny
                wxy = wx * wy
                for zi in range(z0-halfW+1, z0+halfW+1):
                    dz = abs(gz[m] - zi)
                    if dz > W/2: continue
                    zw = zi % Nz
                    grid[xw, yw, zw] += samps[m] * (wxy * kb(dz))

    # crude separable deapodization
    def deapo(n):
        u = np.fft.fftshift(np.arange(n) - n/2.0)
        v = np.abs(u); w = np.ones_like(v, dtype=np.float64)
        mask = v <= W/2
        vv = v[mask]
        x = np.sqrt((1 - (2*vv/W)**2).clip(0,1))
        w[mask] = np.i0(beta * x)
        w[w==0] = 1e-6
        return 1.0 / w

    dx = deapo(Nx); dy = deapo(Ny); dz = deapo(Nz)
    grid *= dx[:,None,None]; grid *= dy[None,:,None]; grid *= dz[None,None,:]

    img_os = np.fft.ifftn(np.fft.ifftshift(grid))
    cx, cy, cz = Nx//2, Ny//2, Nz//2
    half = N//2
    return img_os[cx-half:cx-half+N, cy-half:cy-half+N, cz-half:cz-half+N]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to NPZ with kdata/ktraj/dcf/meta")
    ap.add_argument("--outfile", default="recon_mag.nii.gz", help="Output NIfTI path")
    ap.add_argument("--fov-mm", type=float, default=None, help="FOV in mm (isotropic). If omitted, voxel=1.0mm.")
    ap.add_argument("--normalize", action="store_true", help="Normalize magnitude to [0,1] before saving")
    ap.add_argument("--osf", type=float, default=2.0, help="Oversampling factor for adjoint")
    ap.add_argument("--N", type=int, default=None, help="Target matrix (fallback to meta['N'])")
    args = ap.parse_args()

    dat = np.load(args.npz, allow_pickle=True)
    kdata = dat["kdata"]
    ktraj = dat["ktraj"]
    dcf   = dat["dcf"].astype(np.float64)
    meta  = dat["meta"].item() if "meta" in dat.files else {}

    N = int(args.N if args.N is not None else meta.get("N", 160))
    osf = float(args.osf)

    print(f"kdata: {kdata.shape} {kdata.dtype}")
    print(f"ktraj: {ktraj.shape} {ktraj.dtype}  min/max: {ktraj.min():.6f}/{ktraj.max():.6f}")
    print(f"dcf  : {dcf.shape} float64 mean={dcf.mean():.6f}")
    print(f"N={N}, osf={osf}")

    # Convert trajectory (cycles/FOV) -> radians for NUFFT
    kx = (2*np.pi) * ktraj[:,0]
    ky = (2*np.pi) * ktraj[:,1]
    kz = (2*np.pi) * ktraj[:,2]

    # DCF-weighted samples
    samps = (kdata.astype(np.complex128) * dcf).ravel()

    # Try fast path, else fallback
    img = try_finufft(kx, ky, kz, samps, N=N, osf=osf)
    if img is None:
        print("[info] Falling back to Kaiser–Bessel gridding (preview quality).")
        img = kb_grid_fallback(ktraj, samps, N=N, osf=osf)

    mag = np.abs(img)
    if args.normalize:
        mag = mag / (mag.max() + 1e-12)

    # Affine (voxel size)
    if args.fov_mm is not None:
        vox = float(args.fov_mm) / float(N)
    else:
        vox = 1.0  # default if FOV unknown
        print("[warn] FOV not provided; assuming voxel size = 1.0 mm")
    affine = np.diag([vox, vox, vox, 1.0])

    # Save NIfTI
    nii = nib.Nifti1Image(mag.astype(np.float32), affine)
    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nii, str(out))
    print(f"Saved NIfTI: {out.resolve()}  (voxel size ~ {vox:.3f} mm)")

    # Also save a quick central slice PNG
    z = mag.shape[2] // 2
    plt.figure(figsize=(6,6))
    plt.imshow((mag / (mag.max()+1e-12))[:,:,z].T, origin="lower", cmap="gray")
    plt.title(f"Adjoint preview (slice {z})")
    plt.axis("off")
    plt.tight_layout()
    png = out.with_suffix("").as_posix() + "_preview.png"
    plt.savefig(png, dpi=300)
    plt.close()
    print(f"Saved preview: {png}")

if __name__ == "__main__":
    main()
