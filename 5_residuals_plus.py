#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 17:05:27 2025

@author: alex
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute operator-weighted k-space residuals for UTE3D recons and save:
- Relative residuals (numbers)
- Phase-proxy central slices (for CIC and FINUFFT)
- Residual histograms (|wF(x)-wy|)
- Residual vs. k-radius curves
- Residual arrays to NPZ

Usage example
-------------
python 5_residuals_plus.py \
  --npz /Users/alex/AlexBadea_MyCodes/ute3d/ute3d_N64_S3300_phys.npz \
  --nufft_nii N64_S3300_finufft_cg.nii.gz \
  --preview_nii preview_recon.nii.gz \
  --osf 2.0 --prefix res_phys
  
  python 5_residuals_plus.py   --npz /Users/alex/AlexBadea_MyCodes/ute3d/ute3d_N64_S3300_phys.npz   --nufft_nii N64_S3300_finufft_cg.nii.gz   --preview_nii preview_recon.nii.gz   --osf 2.0 --prefix res_phys
"""
import argparse, numpy as np, nibabel as nib, matplotlib.pyplot as plt
from pathlib import Path

# ---------- FFT helpers ----------
def fft3c(x):  return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x)))
def ifft3c(X): return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(X)))

def center_pad_to(x, shape):
    out = np.zeros(shape, dtype=x.dtype)
    nx, ny, nz = x.shape; Nx, Ny, Nz = shape
    sx = (Nx - nx)//2; sy = (Ny - ny)//2; sz = (Nz - nz)//2
    out[sx:sx+nx, sy:sy+ny, sz:sz+nz] = x
    return out

def center_crop(x, shape):
    Nx, Ny, Nz = x.shape; nx, ny, nz = shape
    sx = (Nx - nx)//2; sy = (Ny - ny)//2; sz = (Nz - nz)//2
    return x[sx:sx+nx, sy:sy+ny, sz:sz+nz]

def ktraj_to_radians(ktraj):
    u = np.ascontiguousarray(ktraj, dtype=np.float64) * (2*np.pi)
    return u[:,0].copy(), u[:,1].copy(), u[:,2].copy()

def norm01(x):
    x = x - x.min()
    m = x.max()
    return x / m if m > 0 else x

# ---------- CIC NUFFT-ish (NumPy) ----------
def _grid_coords(ktraj, N_os):
    gx = ktraj[:,0] * N_os + (N_os/2.0 - 0.5)
    gy = ktraj[:,1] * N_os + (N_os/2.0 - 0.5)
    gz = ktraj[:,2] * N_os + (N_os/2.0 - 0.5)
    return gx, gy, gz

def cic_deposit_numpy(y, ktraj, N_os):
    Nx = Ny = Nz = N_os
    gx, gy, gz = _grid_coords(ktraj, N_os)
    i0x = np.floor(gx).astype(np.int64).clip(0, Nx-1); i1x = np.minimum(i0x+1, Nx-1)
    i0y = np.floor(gy).astype(np.int64).clip(0, Ny-1); i1y = np.minimum(i0y+1, Ny-1)
    i0z = np.floor(gz).astype(np.int64).clip(0, Nz-1); i1z = np.minimum(i0z+1, Nz-1)
    wx = gx - np.floor(gx); wy = gy - np.floor(gy); wz = gz - np.floor(gz)
    w000=(1-wx)*(1-wy)*(1-wz); w100=wx*(1-wy)*(1-wz)
    w010=(1-wx)*wy*(1-wz);     w110=wx*wy*(1-wz)
    w001=(1-wx)*(1-wy)*wz;     w101=wx*(1-wy)*wz
    w011=(1-wx)*wy*wz;         w111=wx*wy*wz

    grid = np.zeros((Nx,Ny,Nz), dtype=np.complex128)
    def add(ix,iy,iz,wt): np.add.at(grid, (ix,iy,iz), y*wt)
    add(i0x,i0y,i0z,w000); add(i1x,i0y,i0z,w100)
    add(i0x,i1y,i0z,w010); add(i1x,i1y,i0z,w110)
    add(i0x,i0y,i1z,w001); add(i1x,i0y,i1z,w101)
    add(i0x,i1y,i1z,w011); add(i1x,i1y,i1z,w111)

    # occupancy normalization
    occ = np.zeros((Nx,Ny,Nz), dtype=np.float64)
    def addw(ix,iy,iz,wt): np.add.at(occ, (ix,iy,iz), wt)
    addw(i0x,i0y,i0z,w000); addw(i1x,i0y,i0z,w100)
    addw(i0x,i1y,i0z,w010); addw(i1x,i1y,i0z,w110)
    addw(i0x,i0y,i1z,w001); addw(i1x,i0y,i1z,w101)
    addw(i0x,i1y,i1z,w011); addw(i1x,i1y,i1z,w111)

    return grid / (occ + 1e-8)

def cic_interpolate_numpy(grid, ktraj, N_os):
    Nx,Ny,Nz = grid.shape
    gx, gy, gz = _grid_coords(ktraj, N_os)
    i0x = np.floor(gx).astype(np.int64).clip(0, Nx-1); i1x = np.minimum(i0x+1, Nx-1)
    i0y = np.floor(gy).astype(np.int64).clip(0, Ny-1); i1y = np.minimum(i0y+1, Ny-1)
    i0z = np.floor(gz).astype(np.int64).clip(0, Nz-1); i1z = np.minimum(i0z+1, Nz-1)
    wx = gx - np.floor(gx); wy = gy - np.floor(gy); wz = gz - np.floor(gz)
    w000=(1-wx)*(1-wy)*(1-wz); w100=wx*(1-wy)*(1-wz)
    w010=(1-wx)*wy*(1-wz);     w110=wx*wy*(1-wz)
    w001=(1-wx)*(1-wy)*wz;     w101=wx*(1-wy)*wz
    w011=(1-wx)*wy*wz;         w111=wx*wy*wz
    g = grid
    return (w000*g[i0x,i0y,i0z] + w100*g[i1x,i0y,i0z] +
            w010*g[i0x,i1y,i0z] + w110*g[i1x,i1y,i0z] +
            w001*g[i0x,i0y,i1z] + w101*g[i1x,i0y,i1z] +
            w011*g[i0x,i1y,i1z] + w111*g[i1x,i1y,i1z])

def make_ops_cic(N, osf, ktraj, dcf):
    N_os = int(round(osf * N))
    w = np.sqrt(np.maximum(dcf.astype(np.float64), 0.0))
    def A(x):
        Xos = fft3c(center_pad_to(x, (N_os,N_os,N_os)))
        y  = cic_interpolate_numpy(Xos, ktraj, N_os)
        return (w * y).astype(np.complex128)
    def AH(y):
        grid = cic_deposit_numpy((w * y).astype(np.complex128), ktraj, N_os)
        xos = ifft3c(grid)
        return center_crop(xos, (N,N,N)).astype(np.complex128)
    return A, AH

# ---------- FINUFFT (version-agnostic wrappers) ----------
_HAS_FINUFFT = False
try:
    import finufft
    _HAS_FINUFFT = True
except Exception:
    _HAS_FINUFFT = False

def _fnufft_type1(xj, yj, zj, cj, ms, mt, mu, isign=+1, eps=1e-9):
    last_err = None
    try:
        return finufft.nufft3d1(xj, yj, zj, cj, ms, mt, mu, isign=isign, eps=eps)
    except Exception as e:
        last_err = e
    try:
        return finufft.nufft3d1(xj, yj, zj, cj, isign, eps, ms, mt, mu)
    except Exception as e:
        last_err = e
    try:
        return finufft.nufft3d1(xj, yj, zj, cj, (ms, mt, mu), isign=isign, eps=eps)
    except Exception as e:
        last_err = e
    try:
        return finufft.nufft3d1(xj, yj, zj, cj, (ms, mt, mu), isign, eps)
    except Exception as e:
        last_err = e
    raise TypeError(f"FINUFFT nufft3d1 signature not recognized: {last_err}")

def _fnufft_type2(xj, yj, zj, fk, isign=+1, eps=1e-9):
    last_err = None
    try:
        return finufft.nufft3d2(xj, yj, zj, fk, isign=isign, eps=eps)
    except Exception as e:
        last_err = e
    try:
        return finufft.nufft3d2(xj, yj, zj, isign, eps, fk)
    except Exception as e:
        last_err = e
    try:
        return finufft.nufft3d2(xj, yj, zj, fk, eps=eps, isign=isign)
    except Exception as e:
        last_err = e
    raise TypeError(f"FINUFFT nufft3d2 signature not recognized: {last_err}")

def make_ops_finufft(N, osf, ktraj, dcf, eps=1e-9):
    if not _HAS_FINUFFT:
        raise RuntimeError("finufft not installed.")
    N_os = int(round(osf * N))
    ms = mt = mu = N_os
    xj, yj, zj = ktraj_to_radians(ktraj)
    w = np.sqrt(np.maximum(dcf.astype(np.float64), 0.0))
    def A(x):
        Xos = fft3c(center_pad_to(x, (N_os,N_os,N_os))).astype(np.complex128, copy=False)
        fk  = np.ascontiguousarray(Xos)
        y   = _fnufft_type2(xj, yj, zj, fk, isign=+1, eps=eps)
        return (w * y).astype(np.complex128)
    def AH(y):
        grid = _fnufft_type1(xj, yj, zj, (w * y).astype(np.complex128),
                             ms, mt, mu, isign=+1, eps=eps)
        xos = ifft3c(grid.reshape(ms, mt, mu))
        return center_crop(xos, (N,N,N)).astype(np.complex128)
    return A, AH

# ---------- Plot helpers ----------
def save_phase_slice(vol_cplx, out_png, title):
    z = vol_cplx.shape[2]//2
    ph = np.angle(vol_cplx[:,:,z])
    plt.figure(figsize=(6,6))
    plt.imshow(ph.T, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi)
    plt.axis("off"); plt.title(title); plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()

def save_hist(arr_abs, out_png, title):
    plt.figure(figsize=(6,5))
    plt.hist(arr_abs, bins=200, log=True)
    plt.xlabel("|weighted residual|"); plt.ylabel("count")
    plt.title(title); plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()

def save_curve(x, y, out_png, title, xlabel, ylabel):
    plt.figure(figsize=(6,5))
    plt.plot(x, y, marker="o", linewidth=1)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Residuals, phase slices, and histograms for UTE3D recons")
    ap.add_argument("--npz", required=True, help="NPZ with kdata, ktraj, dcf, meta")
    ap.add_argument("--nufft_nii", required=False, help="NUFFT-CG magnitude NIfTI")
    ap.add_argument("--preview_nii", required=False, help="Preview CG magnitude NIfTI")
    ap.add_argument("--osf", type=float, default=2.0, help="Oversampling factor used in recon")
    ap.add_argument("--eps", type=float, default=1e-9, help="FINUFFT tolerance")
    ap.add_argument("--prefix", type=str, default="res", help="Output prefix for figures/metrics")
    args = ap.parse_args()

    outp = Path(args.prefix)
    outp.mkdir(parents=True, exist_ok=True)

    z = np.load(args.npz, allow_pickle=True)
    kdata = z["kdata"]; ktraj = z["ktraj"]; dcf = z["dcf"]; meta = z["meta"].item()
    N = int(meta.get("N", 64))
    kmax = float(meta.get("kmax", 0.5))
    print(f"Loaded NPZ: N={N}, samples={kdata.size}, osf={args.osf}, kmax={kmax}")

    # weights and denom
    w = np.sqrt(np.maximum(dcf.astype(np.float64), 0.0))
    wy = w * kdata.astype(np.complex128)
    denom = np.linalg.norm(wy) + 1e-30

    # CIC operator
    A_cic, AH_cic = make_ops_cic(N, args.osf, ktraj, dcf)

    # FINUFFT operator (optional)
    A_fn = AH_fn = None
    if _HAS_FINUFFT:
        A_fn, AH_fn = make_ops_finufft(N, args.osf, ktraj, dcf, eps=args.eps)
    else:
        print("[warn] finufft not available; skipping FINUFFT outputs.")

    metrics_lines = []

    def process_one(tag, mag_path, A, AH):
        if (mag_path is None) or (A is None) or (AH is None):
            print(f"[skip] {tag}: missing inputs/operator.")
            return None

        # magnitude volume
        mag = nib.load(mag_path).get_fdata().astype(np.float64)
        mag = mag / (mag.max() + 1e-12)

        # phase proxy from adjoint of measured data (unweighted y; AH applies w internally)
        ah = AH(kdata.astype(np.complex128))
        save_phase_slice(ah, outp/f"{tag}_phase_proxy.png", f"{tag} phase proxy (central slice)")

        ph = np.angle(ah)
        x = mag * np.exp(1j * ph)

        pred = A(x)        # already weighted by w
        r = pred - wy      # weighted residual
        rel = float(np.linalg.norm(r) / denom)

        # Hist and curves
        absr = np.abs(r).astype(np.float64)
        save_hist(absr, outp/f"{tag}_residual_hist.png", f"{tag} |weighted residual| histogram")

        # residual vs k-radius
        kr = np.linalg.norm(ktraj.astype(np.float64), axis=1)  # in cycles/FOV
        # normalize by kmax if present
        if kmax > 0:
            krn = kr / kmax
        else:
            krn = kr
        # bin into 50 radial bins
        bins = np.linspace(0, krn.max(), 51)
        idx = np.digitize(krn, bins) - 1
        prof = np.zeros(50, dtype=np.float64); cnt = np.zeros(50, dtype=np.int64)
        for i in range(50):
            m = (idx == i)
            if np.any(m): prof[i] = absr[m].mean(); cnt[i] = m.sum()
        centers = 0.5*(bins[:-1] + bins[1:])
        save_curve(centers, prof, outp/f"{tag}_residual_vs_radius.png",
                   f"{tag} residual vs k-radius", "k/kmax", "mean |weighted residual|")

        # metrics summary
        med = float(np.median(absr)); p95 = float(np.percentile(absr, 95)); mx = float(absr.max())
        lines = [
            f"[{tag}] rel_res={rel:.6e}",
            f"[{tag}] median={med:.6e}  p95={p95:.6e}  max={mx:.6e}",
            f"[{tag}] count={absr.size}"
        ]
        print("\n".join(lines))
        metrics_lines.extend(lines)

        # dump arrays
        np.savez(outp/f"{tag}_residuals.npz",
                 residual=r, residual_abs=absr, k_radius=kr, k_radius_norm=krn)

        return rel

    rel_preview = process_one("CIC_preview",
                              args.preview_nii, A_cic, AH_cic)
    rel_nufft = None
    if _HAS_FINUFFT:
        rel_nufft = process_one("FINUFFT_nufft",
                                args.nufft_nii, A_fn, AH_fn)

    # Save aggregated metrics
    (outp/"metrics.txt").write_text("\n".join(metrics_lines) + "\n")
    print(f"\nSaved figures + metrics under: {outp.resolve()}")

if __name__ == "__main__":
    main()
