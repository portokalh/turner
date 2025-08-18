#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a Bruker-style UTE3D radial NPZ and optional preview recon (PNG + NIfTI),
with pluggable simulators and phantom saving.

Outputs
-------
- <outfile>.npz with:
  kdata: (M,) complex64
  ktraj: (M,3) float32  (cycles/FOV)
  dcf  : (M,) float32   (safe radial; mean=1)
  meta : dict

- preview_slice.png         (central slice magnitude, normalized)
- preview_recon.nii.gz      (3D magnitude volume, normalized)
- (optional phantom) <prefix>_mag.png and <prefix>_mag.nii.gz

Simulators (choose with --sim)
------------------------------
- gaussian       : built-in synthetic Gaussian blob phantom
- gaussian_real  : calls your simulate_kdata_gaussian_real(...)
- physical       : calls your simulate_kdata_physical(...)

If a simulator cannot return an analytic phantom, a noise-free **adjoint
reference** is produced by depositing kdata (DCF-weighted) and IFFT.

Usage examples
--------------
# Simple Gaussian, center-out, save phantom
python 1_make_npz.py --sim gaussian --N 64 --spokes 3300 --readout 257 \
  --center_out --pair_dirs --kmax 0.5 --noise_rel 0.0 --fov_mm 20 \
  --outfile ute3d_N64_S3300_gauss.npz \
  --save_phantom --phantom_prefix phantom_N64_S3300

# Use your gaussian_real simulator + CG preview
python 1_make_npz.py --sim gaussian_real --N 64 --spokes 3300 --readout 257 \
  --center_out --pair_dirs --kmax 0.5 --noise_rel 0.0 --fov_mm 20 \
  --recon cg --lambda_l2 1e-4 --lambda_grad 5e-4 --max_iter 15 \
  --outfile ute3d_N64_S3300_cg.npz --save_qa --qa_prefix qa_N64_S3300 \
  --save_phantom --phantom_prefix phantom_greal

# Use your physical simulator
python 1_make_npz.py --sim physical --N 64 --spokes 3300 --readout 257 \
  --center_out --pair_dirs --kmax 0.5 --noise_rel 0.0 --fov_mm 20 \
  --T2star_ms 2.0 --TE_us 80 --dwell_us 8 --df_Hz 25 \
  --outfile ute3d_N64_S3300_phys.npz --save_phantom --phantom_prefix phantom_phys
"""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# ------------------------------ Helpers -------------------------------- #

def gen_fibonacci_directions(n_dirs: int, seed: int = 1234):
    """Approximately uniform directions on the sphere for radial spokes."""
    rnd = np.random.RandomState(seed)
    offset = 2.0 / n_dirs
    increment = np.pi * (3.0 - np.sqrt(5.0))
    dirs = np.zeros((n_dirs, 3), dtype=np.float64)
    for i in range(n_dirs):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(max(1 - y * y, 0.0))
        phi = (i % n_dirs) * increment + 0.1 * rnd.uniform(-1, 1)
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        dirs[i] = [x, y, z]
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    return dirs

def gen_3d_radial_ktraj(spokes: int, readout: int, kmax: float,
                        units: str = "cycles/FOV", center_out: bool = False,
                        pair_dirs: bool = False, seed: int = 1234):
    """
    Generate 3D radial trajectory in 'cycles/FOV'.
      - symmetric:  t in [-kmax, +kmax]
      - center-out: t in [0, +kmax]  (typical UTE)
    If pair_dirs=True with center_out, we emit both ±d rays (doubling directions).

    Returns:
      ktraj: (M,3) float32   where M = n_dirs_eff * readout
      n_dirs_eff: effective directions used
    """
    base_dirs = gen_fibonacci_directions(spokes, seed=seed)
    if center_out and pair_dirs:
        dirs = np.vstack([base_dirs, -base_dirs])
        n_dirs_eff = 2 * spokes
    else:
        dirs = base_dirs
        n_dirs_eff = spokes

    t0 = 0.0 if center_out else -kmax
    t = np.linspace(t0, kmax, readout, dtype=np.float64)

    ktraj = np.empty((n_dirs_eff * readout, 3), dtype=np.float64)
    idx = 0
    for d in dirs:
        ktraj[idx:idx+readout, :] = np.outer(t, d)  # (readout,3)
        idx += readout
    return ktraj.astype(np.float32), n_dirs_eff

def compute_dcf_safe(ktraj: np.ndarray):
    """
    'Safe' radial DCF ~ |k|^2 with small floor near DC; normalized to mean=1.
    Works for center-out or symmetric.
    """
    r2 = np.sum(ktraj.astype(np.float64)**2, axis=1)
    r2 += 1e-6 * np.max(r2)  # small floor
    dcf = r2.astype(np.float64)
    m = np.mean(dcf) if dcf.size else 1.0
    return (dcf / (m if m > 0 else 1.0)).astype(np.float32)

def sha1_of_array(x: np.ndarray) -> str:
    h = hashlib.sha1(); h.update(x.tobytes()); return h.hexdigest()

# ---------- CIC deposit/interpolate (NumPy, vectorized) ---------- #

def _grid_coords(ktraj, N_os):
    gx = ktraj[:,0] * N_os + (N_os/2.0 - 0.5)
    gy = ktraj[:,1] * N_os + (N_os/2.0 - 0.5)
    gz = ktraj[:,2] * N_os + (N_os/2.0 - 0.5)
    return gx, gy, gz

def cic_deposit_numpy(y, ktraj, N_os):
    """Deposit nonuniform samples y to an N_os^3 grid using trilinear kernel."""
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

    # occupancy normalize to reduce density bias
    occ = np.zeros((Nx,Ny,Nz), dtype=np.float64)
    def addw(ix,iy,iz,wt): np.add.at(occ, (ix,iy,iz), wt)
    addw(i0x,i0y,i0z,w000); addw(i1x,i0y,i0z,w100)
    addw(i0x,i1y,i0z,w010); addw(i1x,i1y,i0z,w110)
    addw(i0x,i0y,i1z,w001); addw(i1x,i0y,i1z,w101)
    addw(i0x,i1y,i1z,w011); addw(i1x,i1y,i1z,w111)

    return grid / (occ + 1e-8)

def cic_interpolate_numpy(grid, ktraj, N_os):
    """Interpolate values from a grid at nonuniform ktraj (trilinear)."""
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
    vals = (w000*g[i0x,i0y,i0z] + w100*g[i1x,i0y,i0z] +
            w010*g[i0x,i1y,i0z] + w110*g[i1x,i1y,i0z] +
            w001*g[i0x,i0y,i1z] + w101*g[i1x,i0y,i1z] +
            w011*g[i0x,i1y,i1z] + w111*g[i1x,i1y,i1z])
    return vals

# ---------- Preview recon (adjoint & CG on NumPy CIC) ---------- #

def fft3c(x):  return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x)))
def ifft3c(X): return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(X)))

def center_crop(x, N):
    Nx,Ny,Nz = x.shape; sx=(Nx-N)//2; sy=(Ny-N)//2; sz=(Nz-N)//2
    return x[sx:sx+N, sy:sy+N, sz:sz+N]

def adjoint_preview_image(kdata, ktraj, dcf, N, osf=2.0):
    N_os = int(round(osf * N))
    y = (kdata.astype(np.complex128) * dcf.astype(np.float64))
    grid = cic_deposit_numpy(y, ktraj, N_os)
    img_os = ifft3c(grid)
    return center_crop(img_os, N)

def laplacian3d_np(x):
    xr, xi = x.real, x.imag
    def L(u):
        up = np.pad(u, ((1,1),(1,1),(1,1)), mode='edge')
        return (up[2:,1:-1,1:-1] + up[:-2,1:-1,1:-1] +
                up[1:-1,2:,1:-1] + up[1:-1,:-2,1:-1] +
                up[1:-1,1:-1,2:] + up[1:-1,1:-1,:-2] - 6.0*u)
    return L(xr).astype(np.complex128) + 1j*L(xi).astype(np.complex128)

def make_ops_cic(N, osf, ktraj, dcf):
    N_os = int(round(osf * N))
    w = np.sqrt(np.maximum(dcf.astype(np.float64), 0.0))
    def A(x):
        Xos = fft3c(np.pad(x, (( (N_os-N)//2, ),)*3, mode='constant'))
        y  = cic_interpolate_numpy(Xos, ktraj, N_os)
        return (w * y).astype(np.complex128)
    def AH(y):
        grid = cic_deposit_numpy((w * y).astype(np.complex128), ktraj, N_os)
        xos = ifft3c(grid)
        return center_crop(xos, N).astype(np.complex128)
    return A, AH

def cg_tikhonov_numpy(A, AH, y, lam_l2=0.0, lam_grad=0.0, N=64, max_iter=15, verbose=True):
    b = AH(y)
    x = np.zeros_like(b)
    def N_op(v): return AH(A(v)) + lam_l2*v + lam_grad*laplacian3d_np(v)
    r = b - N_op(x); p = r.copy()
    rs_old = np.vdot(r.ravel(), r.ravel())
    for it in range(1, max_iter+1):
        Ap = N_op(p)
        alpha = rs_old / (np.vdot(p.ravel(), Ap.ravel()) + 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.vdot(r.ravel(), r.ravel())
        if verbose: print(f"[cg] it={it:02d} |r|={np.sqrt(rs_new.real):.3e}")
        if np.sqrt(rs_new.real) < 1e-6: break
        p = r + (rs_new/rs_old) * p
        rs_old = rs_new
    return x

# ---------- Built-in 'gaussian' simulator & phantom ---------- #


def simulate_kdata_physical(ktraj,
                            readout: int,
                            T2star_ms: float = 2.0,
                            TE_us: float = 80.0,
                            dwell_us: float = 8.0,
                            df_Hz: float = 25.0,
                            noise_rel: float = 0.0,
                            seed: int = 1234):
    """
    Minimal 'physical' UTE sim:
      - start center-out at TE, per-sample dwell
      - exponential T2* decay
      - global off-resonance frequency df_Hz phase rotation
      - underlying spatial content: reuse 'gaussian' phantom in k-space
        (so we get a reasonable spatial target)
    """
    import numpy as np

    # 1) Base spatial content (same as gaussian sim, but noise-free here)
    kdata_base, params = simulate_kdata_gaussian(ktraj, seed=seed, noise_rel=0.0, return_params=True)

    # 2) Time axis per readout sample (seconds)
    TE_s = float(TE_us) * 1e-6
    dt_s = float(dwell_us) * 1e-6
    t_readout = TE_s + np.arange(readout, dtype=np.float64) * dt_s  # shape (readout,)

    # 3) Build per-sample temporal kernel for the whole acquisition
    M = ktraj.shape[0]
    n_spokes = int(np.ceil(M / readout))
    if n_spokes * readout != M:
        # if not divisible, clip to full spokes (shouldn't happen with this script)
        usable = n_spokes * readout
        kdata_base = kdata_base[:usable]
        ktraj = ktraj[:usable, :]
        M = usable

    # repeat time kernel for each spoke
    t_all = np.tile(t_readout, n_spokes)[:M]  # (M,)

    # 4) T2* decay + off-resonance phase
    T2star_s = float(T2star_ms) * 1e-3
    decay = np.exp(-t_all / max(T2star_s, 1e-12))              # (M,)
    phase = np.exp(1j * (2.0 * np.pi * float(df_Hz)) * t_all)  # (M,)

    kdata = kdata_base.astype(np.complex128) * decay * phase

    # 5) Add complex white noise relative to current std
    if noise_rel > 0.0:
        rng = np.random.RandomState(seed)
        sig = np.std(kdata) if np.std(kdata) > 0 else 1.0
        sigma = noise_rel * sig
        noise = (rng.randn(M) + 1j * rng.randn(M)) * (sigma / np.sqrt(2.0))
        kdata = kdata + noise

    return kdata.astype(np.complex64)




def simulate_kdata_gaussian(ktraj: np.ndarray, seed: int = 42, noise_rel: float = 0.0,
                            return_params: bool = False):
    """
    Synthetic k-space from a sum of Gaussian blobs in image space.
    """
    rng = np.random.RandomState(seed)
    n = ktraj.shape[0]
    K = 5
    amps   = rng.randn(K) * 0.8
    phases = rng.rand(K) * 2 * np.pi
    centers = rng.randn(K, 3) * 0.15     # FOV units
    sigma_k = 0.8

    kr = np.linalg.norm(ktraj.astype(np.float64), axis=1)
    signal = np.zeros(n, dtype=np.complex128)
    for a, p, c in zip(amps, phases, centers):
        phase = -2j * np.pi * (ktraj @ c.astype(np.float32))
        env = np.exp(- (kr * sigma_k) ** 2)
        signal += a * env * np.exp(phase + 1j * p)

    sigma = noise_rel * np.std(signal) if np.std(signal) > 0 else noise_rel
    noise = (rng.randn(n) + 1j * rng.randn(n)) * sigma / np.sqrt(2)
    signal = (signal + noise).astype(np.complex64)

    if return_params:
        params = {"amps": amps, "phases": phases, "centers": centers,
                  "sigma_k": float(sigma_k), "seed": int(seed)}
        return signal, params
    return signal

def build_phantom_from_gaussian_params(N: int, params: dict):
    amps   = np.asarray(params["amps"])
    phases = np.asarray(params["phases"])
    centers= np.asarray(params["centers"])
    sigma_k = float(params.get("sigma_k", 0.8))
    sigma_img = 1.0 / (2.0 * np.pi * sigma_k)
    ax = (np.arange(N) - N/2) / N
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    img = np.zeros((N,N,N), dtype=np.complex128)
    for a, p, c in zip(amps, phases, centers):
        r2 = (X-c[0])**2 + (Y-c[1])**2 + (Z-c[2])**2
        img += a * np.exp(-r2/(2.0*sigma_img**2)) * np.exp(1j*p)
    return img

# ---------- Optional: use your external simulators if present ---------- #

def call_simulator_if_available(name):
    """Return a callable if a function exists in the global namespace; else None."""
    return globals().get(name, None)

# ------------------------------ Main ----------------------------------- #

def main():
    p = argparse.ArgumentParser(description="Generate/pack UTE3D radial NPZ and preview recon.")
    # Build/trajectory parameters
    p.add_argument("--N", type=int, default=160, help="Matrix size (isotropic).")
    p.add_argument("--spokes", type=int, default=1200, help="Number of radial spokes (base).")
    p.add_argument("--readout", type=int, default=256, help="Readout samples per spoke.")
    p.add_argument("--kmax", type=float, default=0.5, help="Max k in cycles/FOV (e.g., 0.5).")
    p.add_argument("--center_out", action="store_true", help="Use center-out (0..kmax) trajectory.")
    p.add_argument("--pair_dirs", action="store_true", help="For center-out, use ±d directions (2x spokes).")
    p.add_argument("--os_forward", type=float, default=2.0, help="Oversampling for preview/recon.")
    p.add_argument("--units", type=str, default="cycles/FOV", help="Trajectory units.")
    # Sim choice + physics
    p.add_argument("--sim", choices=["gaussian","gaussian_real","physical"], default="gaussian",
                   help="Simulator to generate k-space.")
    p.add_argument("--T2star_ms", type=float, default=2.0)
    p.add_argument("--TE_us", type=float, default=80.0)
    p.add_argument("--dwell_us", type=float, default=8.0)
    p.add_argument("--df_Hz", type=float, default=25.0)
    p.add_argument("--noise_rel", type=float, default=0.0)
    p.add_argument("--fov_mm", type=float, default=None, help="Isotropic FOV (mm) for NIfTI voxel size.")
    # Recon preview
    p.add_argument("--recon", choices=["none","adjoint","cg"], default="adjoint",
                   help="Preview reconstruction to save.")
    p.add_argument("--lambda_l2", type=float, default=1e-4)
    p.add_argument("--lambda_grad", type=float, default=5e-4)
    p.add_argument("--max_iter", type=int, default=15)
    # Phantom saving
    p.add_argument("--save_phantom", action="store_true", help="Save phantom magnitude (PNG+NIfTI) if available.")
    p.add_argument("--phantom_prefix", type=str, default="phantom", help="Prefix for phantom outputs.")
    # QA (optional PSF sampling kernel volumes)
    p.add_argument("--save_qa", action="store_true", help="Save QA PSF sampling/kernel NIfTIs.")
    p.add_argument("--qa_prefix", type=str, default="qa_ute3d")
    # Optional input arrays
    p.add_argument("--kdata_npy", type=str, help="Path to existing kdata.npy (complex64).")
    p.add_argument("--ktraj_npy", type=str, help="Path to existing ktraj.npy (float32, [M,3]).")
    p.add_argument("--dcf_npy", type=str, help="Path to existing dcf.npy (float32). If omitted, computed.")
    # Output path
    p.add_argument("--outfile", type=str, required=True, help="Output NPZ file path.")
    # Reproducibility
    p.add_argument("--seed", type=int, default=1234, help="Seed for synthetic generation.")
    args = p.parse_args()

    # ---- Load or build ktraj ----
    if args.ktraj_npy:
        ktraj = np.load(args.ktraj_npy).astype(np.float32, copy=False)
        assert ktraj.ndim == 2 and ktraj.shape[1] == 3, "ktraj must be (M,3)"
        n_dirs_eff = ktraj.shape[0] // args.readout
    else:
        ktraj, n_dirs_eff = gen_3d_radial_ktraj(args.spokes, args.readout, args.kmax,
                                                units=args.units, center_out=args.center_out,
                                                pair_dirs=args.pair_dirs, seed=args.seed)

    total_nominal = (2*args.spokes if (args.center_out and args.pair_dirs) else args.spokes) * args.readout
    if ktraj.shape[0] != total_nominal:
        print(f"[warn] ktraj length ({ktraj.shape[0]}) != spokes*readout ({total_nominal})")

    # ---- Load or compute DCF ----
    if args.dcf_npy:
        dcf = np.load(args.dcf_npy).astype(np.float32, copy=False)
        assert dcf.ndim == 1 and dcf.shape[0] == ktraj.shape[0], "dcf must be shape (M,)"
        dcf = (dcf / (dcf.mean() if dcf.mean()>0 else 1.0)).astype(np.float32)
        has_dc = True
    else:
        dcf = compute_dcf_safe(ktraj); has_dc = True

    # ---- Load or build kdata (+ phantom via simulator dispatcher) ----
    phantom = None
    sim_meta = {"name": "external_kdata"} if args.kdata_npy else {}
    if args.kdata_npy:
        kdata = np.load(args.kdata_npy).astype(np.complex64, copy=False)
        assert kdata.ndim == 1 and kdata.shape[0] == ktraj.shape[0], "kdata length mismatch"
    else:
        if args.sim == "gaussian":
            kdata, params = simulate_kdata_gaussian(ktraj, seed=args.seed,
                                                    noise_rel=args.noise_rel, return_params=True)
            phantom = build_phantom_from_gaussian_params(args.N, params)
            sim_meta = {"name":"gaussian", **{k:(v.tolist() if isinstance(v,np.ndarray) else v)
                                              for k,v in params.items()}}
        elif args.sim == "gaussian_real":
            fn = call_simulator_if_available("simulate_kdata_gaussian_real")
            if fn is None:
                raise RuntimeError("simulate_kdata_gaussian_real not found in this script. "
                                   "Define it or switch --sim.")
            out = fn(ktraj, seed=args.seed, noise_rel=args.noise_rel, return_params=True)
            # Accept (kdata, params) or (kdata, phantom, params)
            if isinstance(out, tuple) and len(out) == 3:
                kdata, maybe_phantom, params = out
                phantom = maybe_phantom
            else:
                kdata, params = out
                # Try to build phantom from params if fields exist; else leave None
                try:
                    phantom = build_phantom_from_gaussian_params(args.N, params)
                except Exception:
                    phantom = None
            sim_meta = {"name":"gaussian_real", **{k:(v.tolist() if isinstance(v,np.ndarray) else v)
                                                   for k,v in params.items()}}
        elif args.sim == "physical":
            fn = call_simulator_if_available("simulate_kdata_physical")
            if fn is None:
                raise RuntimeError("simulate_kdata_physical not found in this script. Define it or switch --sim.")
                
            kdata = fn(ktraj, readout=args.readout,
                          T2star_ms=args.T2star_ms, TE_us=args.TE_us,
                          dwell_us=args.dwell_us, df_Hz=args.df_Hz,
                          noise_rel=args.noise_rel, seed=args.seed)
    
            # Build a noise-free reference phantom (for saving magnitude image)
            kdata_nf = fn(ktraj, readout=args.readout,
                          T2star_ms=args.T2star_ms, TE_us=args.TE_us,
                          dwell_us=args.dwell_us, df_Hz=args.df_Hz,
                          noise_rel=0.0, seed=args.seed)

                
            '''          
            kdata = fn(ktraj, T2star_ms=args.T2star_ms, TE_us=args.TE_us,
                       dwell_us=args.dwell_us, df_Hz=args.df_Hz,
                       noise_rel=args.noise_rel, seed=args.seed)
            # Build a *reference* phantom via noise-free adjoint
            
            
            
            
            kdata_nf = fn(ktraj, T2star_ms=args.T2star_ms, TE_us=args.TE_us,
                          dwell_us=args.dwell_us, df_Hz=args.df_Hz,
                          noise_rel=0.0, seed=args.seed)
            
            '''
            phantom = adjoint_preview_image(kdata_nf, ktraj, dcf, args.N, osf=args.os_forward)
            sim_meta = {"name":"physical", "T2star_ms":float(args.T2star_ms),
                        "TE_us":float(args.TE_us), "dwell_us":float(args.dwell_us),
                        "df_Hz":float(args.df_Hz)}
        else:
            raise ValueError(f"Unknown --sim {args.sim}")

        if kdata.dtype != np.complex64:
            kdata = kdata.astype(np.complex64)

    # ---- Sanity ----
    if ktraj.shape[0] != kdata.shape[0] or dcf.shape[0] != kdata.shape[0]:
        raise ValueError("Size mismatch among kdata, ktraj, and dcf.")

    # ---- Metadata ----
    meta = {
        "N": int(args.N),
        "spokes": int(args.spokes),
        "readout": int(args.readout),
        "kmax": float(args.kmax),
        "os_forward": float(args.os_forward),
        "T2star_ms": float(args.T2star_ms),
        "TE_us": float(args.TE_us),
        "dwell_us": float(args.dwell_us),
        "df_Hz": float(args.df_Hz),
        "noise_rel": float(args.noise_rel),
        "ktraj_units": f"cycles/FOV in [{'0,' if args.center_out else f'-{args.kmax}, '}{args.kmax}]",
        "dcf": "safe radial DCF (mean=1)",
        "sha1_kdata": sha1_of_array(kdata),
        "sha1_ktraj": sha1_of_array(ktraj),
        "sha1_dcf": sha1_of_array(dcf),
        "center_out": bool(args.center_out),
        "pair_dirs": bool(args.pair_dirs),
        "n_dirs_effective": int(n_dirs_eff),
        "use_physical": (args.sim == "physical"),
        "simulator": sim_meta
    }

    # ---- Save NPZ ----
    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out), kdata=kdata, ktraj=ktraj.astype(np.float32),
                       dcf=dcf.astype(np.float32), meta=meta)
    print(f"has_dc: {has_dc} | center_out: {args.center_out} | pair_dirs: {args.pair_dirs}")
    print(f"kdata: {kdata.shape} {kdata.dtype}")
    print(f"ktraj: {ktraj.shape} {ktraj.dtype} min/max: {ktraj.min():.6f} {ktraj.max():.6f}")
    print(f"dcf  : {dcf.shape} {dcf.dtype}  mean: {dcf.mean():.6f}  min/max: {dcf.min():.6f} {dcf.max():.6f}")
    print("meta :", json.dumps(meta, indent=2, sort_keys=True))
    print(f"\nSaved: {out.resolve()}")

    # ---- Preview recon ----
    if args.recon != "none":
        if args.recon == "adjoint":
            img = adjoint_preview_image(kdata, ktraj, dcf, args.N, osf=args.os_forward)
        else:
            A, AH = make_ops_cic(args.N, args.os_forward, ktraj, dcf)
            img = cg_tikhonov_numpy(A, AH, kdata.astype(np.complex128),
                                    lam_l2=args.lambda_l2, lam_grad=args.lambda_grad,
                                    N=args.N, max_iter=args.max_iter, verbose=True)
        mag = np.abs(img); mag /= (mag.max() + 1e-12)
        # PNG (central slice)
        z = args.N // 2
        plt.figure(figsize=(6,6))
        plt.imshow(mag[:, :, z].T, cmap="gray", origin="lower")
        plt.axis("off")
        ttl = "CG" if args.recon=="cg" else "Adjoint"
        plt.title(f"Preview ({ttl}; slice {z})")
        plt.tight_layout()
        plt.savefig("preview_slice.png", dpi=300); plt.close()
        print("Saved preview_slice.png")
        # NIfTI
        if args.fov_mm is not None:
            vox = float(args.fov_mm) / float(args.N)
        else:
            vox = 1.0
            print("[warn] FOV not provided; using voxel size 1.0 mm")
        affine = np.diag([vox, vox, vox, 1.0])
        nib.save(nib.Nifti1Image(mag.astype(np.float32), affine), "preview_recon.nii.gz")
        if args.recon == "cg":
            print("Saved preview_slice.png and preview_recon.nii.gz (CG)")
        else:
            print("Saved preview_slice.png and preview_recon.nii.gz (Adjoint)")

    # ---- Save phantom if available ----
    if args.save_phantom and (phantom is not None):
        mag = np.abs(phantom); mag /= (mag.max() + 1e-12)
        z = args.N // 2
        plt.figure(figsize=(6,6))
        plt.imshow(mag[:, :, z].T, cmap="gray", origin="lower")
        plt.axis("off")
        plt.title(f"Phantom magnitude (slice {z})")
        plt.tight_layout()
        plt.savefig(f"{args.phantom_prefix}_mag.png", dpi=300); plt.close()
        vox = (float(args.fov_mm)/float(args.N)) if (args.fov_mm is not None) else 1.0
        if args.fov_mm is None:
            print("[warn] FOV not provided; phantom NIfTI voxel size set to 1.0 mm")
        affine = np.diag([vox, vox, vox, 1.0])
        nib.save(nib.Nifti1Image(mag.astype(np.float32), affine),
                 f"{args.phantom_prefix}_mag.nii.gz")
        print(f"Saved {args.phantom_prefix}_mag.png and {args.phantom_prefix}_mag.nii.gz")

    # ---- Optional QA (PSF sampling/kernel) ----
    if args.save_qa:
        N_os = int(round(args.os_forward * args.N))
        # Sampling PSF ~ adjoint of ones
        one = np.ones(ktraj.shape[0], dtype=np.complex128)
        psf_samp = adjoint_preview_image(one, ktraj, dcf, args.N, osf=args.os_forward)
        psf_kern = ifft3c(np.ones((N_os,N_os,N_os), dtype=np.complex128))
        affine = np.eye(4, dtype=float)
        nib.save(nib.Nifti1Image(np.abs(psf_samp).astype(np.float32), affine),
                 f"{args.qa_prefix}_psf_sampling.nii.gz")
        nib.save(nib.Nifti1Image(np.abs(psf_kern).astype(np.float32), affine),
                 f"{args.qa_prefix}_psf_kernel.nii.gz")
        print(f"Saved {args.qa_prefix}_psf_sampling.nii.gz and {args.qa_prefix}_psf_kernel.nii.gz")

if __name__ == "__main__":
    main()
