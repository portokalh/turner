#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a Bruker-style UTE3D radial NPZ and optional preview recon (PNG + NIfTI).

Outputs
-------
- <outfile>.npz with:
  kdata: (M,) complex64
  ktraj: (M,3) float32  (cycles/FOV)
  dcf  : (M,) float32   (density compensation, mean=1)
  meta : dict

- preview_slice.png         (central slice magnitude, normalized)
- preview_recon.nii.gz      (3D magnitude volume, normalized)

Optional QA (with --save_qa)
----------------------------
- <qa_prefix>_kspace_os.npy       (complex64, oversampled Cartesian k-space; DC centered)
- <qa_prefix>_kspace_mask.npy     (float32, binary occupancy mask on the oversampled grid)
- <qa_prefix>_psf_sampling.nii.gz (float32, |IFFT(occupancy)|, peak-normalized)
- <qa_prefix>_psf_kernel.nii.gz   (float32, |IFFT(CIC kernel-sum)|, peak-normalized)

Quick sanity checks (should look clean)
---------------------------------------
# Smaller matrix, lots of spokes, DC captured, no noise
# (Produces clean Gaussian blobs; not snow.)
#

python 1_make_npz.py --N 64 --spokes 1200 --readout 257 \
  --center_out --pair_dirs --kmax 0.5 --noise_rel 0.0 \
  --outfile ute3d_small.npz
  
python 1_make_npz.py --N 96 --spokes 1200 --readout 257 \
  --center_out --pair_dirs --kmax 0.5 --noise_rel 0.0 \
  --outfile ute3d_radial.npz --save_qa --qa_prefix qa_ute3d

# python 1_make_npz.py --N 96 --spokes 12000 --readout 257 \
#   --center_out --pair_dirs --kmax 0.5 --noise_rel 0.0 \
#   --outfile ute3d_radial.npz --save_qa --qa_prefix qa_ute3d
#
# Moderate matrix, fewer spokes (some streaks), with T2* and off-resonance
#
# python 1_make_npz.py --N 128 --spokes 8000 --readout 257 \
#   --center_out --pair_dirs --kmax 0.5 --use_physical \
#   --T2star_ms 2.0 --df_Hz 25 --outfile ute3d_phys.npz \
#   --save_qa --qa_prefix qa_phys
"""
import argparse
import json
import hashlib
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
    dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
    return dirs


def gen_3d_radial_ktraj(spokes: int, readout: int, kmax: float,
                        units: str = "cycles/FOV", center_out: bool = False,
                        pair_dirs: bool = False, seed: int = 1234):
    """
    Generate 3D radial trajectory in 'cycles/FOV'.
      - symmetric:  t in [-kmax, +kmax]  (use ODD readout to hit k=0)
      - center-out: t in [0, +kmax]      (includes k=0)

    If center_out and pair_dirs=True: duplicate directions with their negatives (±d)
    so each spoke has both half-lines. If you leave pair_dirs=False, you will
    only sample half of k-space (intentional for some flows, but usually not).
    """
    dirs = gen_fibonacci_directions(spokes, seed=seed)

    if center_out and pair_dirs:
        print("[info] pairing directions (±d) for center-out.")
        dirs = np.vstack([dirs, -dirs])

    t = np.linspace(0.0, kmax, readout, dtype=np.float64) if center_out \
        else np.linspace(-kmax, kmax, readout, dtype=np.float64)

    ktraj = np.empty((dirs.shape[0] * readout, 3), dtype=np.float64)
    idx = 0
    for d in dirs:
        ktraj[idx:idx+readout, :] = np.outer(t, d)
        idx += readout
    return ktraj.astype(np.float32)



def compute_dcf_safe(ktraj: np.ndarray, p: float = 2.0, rmin: float = 1e-3):
    """
    Simple radial DCF ~ max(|k|, rmin)^p, normalized to mean=1.
    p=2 recovers |k|^2 heuristic; rmin avoids vanishing at DC.
    """
    r = np.linalg.norm(ktraj.astype(np.float64), axis=1)
    dcf = np.power(np.maximum(r, rmin), p)
    m = dcf.mean() if dcf.size > 0 else 1.0
    return (dcf / (m if m > 0 else 1.0)).astype(np.float32)


# ------------------------------ Simulators ----------------------------- #

def simulate_kdata_gaussians_real(ktraj: np.ndarray,
                                  n_blobs: int = 4,
                                  sigma: float = 0.12,  # in FOV units
                                  seed: int = 42,
                                  noise_rel: float = 0.0):
    """
    Analytic FT of a sum of real, positive 3D Gaussian blobs.
    Ensures conjugate-symmetric k-space → coherent adjoint image.
    ktraj in cycles/FOV; centers in [-0.3, 0.3] FOV units.
    """
    rng = np.random.RandomState(seed)
    M = ktraj.shape[0]
    # real, positive amplitudes and real-space centers
    amps = 0.5 + rng.rand(n_blobs) * 0.8
    centers = rng.uniform(-0.3, 0.3, size=(n_blobs, 3))

    # FT{Gaussian} = const * exp(-2*pi^2*sigma^2*||k||^2) * exp(-i*2*pi*k·x0)
    k = ktraj.astype(np.float64)
    kr2 = np.sum(k*k, axis=1)                                         # ||k||^2
    env = np.exp(- (2*np.pi*sigma)**2 * kr2)                          # radial envelope

    sig = np.zeros(M, dtype=np.complex128)
    for a, c in zip(amps, centers):
        sig += a * env * np.exp(-2j*np.pi * (k @ c.astype(np.float64)))

    # optional noise
    if noise_rel > 0:
        sd = noise_rel * (np.std(sig) if np.std(sig)>0 else 1.0)
        sig += (rng.randn(M) + 1j*rng.randn(M)) * sd / np.sqrt(2)

    return sig.astype(np.complex64)


def simulate_kdata_physical(ktraj, spokes, readout,
                            T2star_ms=2.0, TE_us=80.0, dwell_us=8.0,
                            df_Hz=25.0, seed=42, noise_rel=0.01):
    """
    Synthetic object (Gaussian blobs) + readout-time effects:
    exp(-t/T2*) decay and off-resonance phase.
    """
    rng = np.random.RandomState(seed)
    n = ktraj.shape[0]
    # base object: reuse real Gaussian phantom so image stays coherent
    base = simulate_kdata_gaussians_real(ktraj, seed=seed, noise_rel=0.0).astype(np.complex128)

    # time along readout (same for all spokes)
    t = (TE_us + np.arange(readout)*dwell_us) * 1e-6  # seconds
    t = np.tile(t, spokes)

    # T2* and off-resonance
    T2s = max(T2star_ms, 0.1) * 1e-3
    signal = base * np.exp(-t/T2s) * np.exp(-1j*2*np.pi*df_Hz*t)

    if noise_rel > 0:
        sigma = noise_rel * (np.std(signal) if np.std(signal) > 0 else 1.0)
        noise = (rng.randn(n) + 1j*rng.randn(n)) * (sigma/np.sqrt(2))
        signal = signal + noise
    return signal.astype(np.complex64)


# ------------------------------ Utilities ----------------------------- #

def sha1_of_array(x: np.ndarray) -> str:
    h = hashlib.sha1(); h.update(x.tobytes()); return h.hexdigest()


def print_summary(kdata, ktraj, dcf, meta):
    print(f"kdata: {kdata.shape} {kdata.dtype}")
    print(f"ktraj: {ktraj.shape} {ktraj.dtype} min/max: {ktraj.min():.6f} {ktraj.max():.6f}")
    print(f"dcf  : {dcf.shape} {dcf.dtype}  mean: {dcf.mean():.6f}  min/max: {dcf.min():.6f} {dcf.max():.6f}")
    meta_copy = dict(meta); meta_copy['dcf'] = 'safe radial DCF (mean=1)'
    print("meta :", json.dumps(meta_copy, indent=2, sort_keys=True))


# ---------- Preview recon (CIC gridding; weight-normalized) ---------- #

def preview_recon_cic(kdata, ktraj, dcf, N, osf=2.0, fov_mm=None,
                      png_path="preview_slice.png", nii_path="preview_recon.nii.gz",
                      save_qa=False, qa_prefix="qa"):
    """
    3D tri-linear (CIC) gridding onto an oversampled Cartesian grid, weight-normalized.
    ktraj in cycles/FOV ~ [-0.5, 0.5]. Denominator uses kernel-sum ONLY.
    """
    eps = 1e-8
    N_os = int(round(osf * N))
    Nx = Ny = Nz = N_os

    # Map k (cycles/FOV) -> continuous grid coords u in [0, N_os-1], DC at center voxel
    u = ktraj.astype(np.float64) * N_os + N_os/2.0 - 0.5   # (M,3)

    # Keep only samples whose 8-neighborhood intersects the grid
    mask = (u[:,0] >= -1) & (u[:,0] <= Nx) & \
           (u[:,1] >= -1) & (u[:,1] <= Ny) & \
           (u[:,2] >= -1) & (u[:,2] <= Nz)
    print("samples kept after clip:", int(mask.sum()), "/", ktraj.shape[0])

    u  = u[mask]
    d  = kdata.astype(np.complex128)[mask]
    sw = dcf.astype(np.float64)[mask]                   # positive sample weights (DCF)

    # fractional parts and integer bases
    wx = u[:,0] - np.floor(u[:,0]); wy = u[:,1] - np.floor(u[:,1]); wz = u[:,2] - np.floor(u[:,2])
    i0x = np.clip(np.floor(u[:,0]).astype(np.int64), 0, Nx-1); i1x = np.clip(i0x+1, 0, Nx-1)
    i0y = np.clip(np.floor(u[:,1]).astype(np.int64), 0, Ny-1); i1y = np.clip(i0y+1, 0, Ny-1)
    i0z = np.clip(np.floor(u[:,2]).astype(np.int64), 0, Nz-1); i1z = np.clip(i0z+1, 0, Nz-1)

    # 8 trilinear (CIC) kernel weights
    w000 = (1-wx)*(1-wy)*(1-wz); w100 = wx*(1-wy)*(1-wz)
    w010 = (1-wx)*wy*(1-wz);     w110 = wx*wy*(1-wz)
    w001 = (1-wx)*(1-wy)*wz;     w101 = wx*(1-wy)*wz
    w011 = (1-wx)*wy*wz;         w111 = wx*wy*wz

    grid_r = np.zeros((Nx,Ny,Nz), np.float64)
    grid_i = np.zeros((Nx,Ny,Nz), np.float64)
    wgrid  = np.zeros((Nx,Ny,Nz), np.float64)  # DENOM: kernel-sum only

    # numerator uses data * DCF; denominator uses kernel only
    s = d * sw

    def _acc(ix,iy,iz, kernw):
        np.add.at(grid_r, (ix,iy,iz), (s.real * kernw))
        np.add.at(grid_i, (ix,iy,iz), (s.imag * kernw))
        np.add.at(wgrid,  (ix,iy,iz), (kernw))

    _acc(i0x,i0y,i0z, w000); _acc(i1x,i0y,i0z, w100)
    _acc(i0x,i1y,i0z, w010); _acc(i1x,i1y,i0z, w110)
    _acc(i0x,i0y,i1z, w001); _acc(i1x,i0y,i1z, w101)
    _acc(i0x,i1y,i1z, w011); _acc(i1x,i1y,i1z, w111)

    # Weight-normalized complex grid (DC is at center)
    grid = (grid_r + 1j*grid_i) / (wgrid + eps)

    # Save QA k-space + mask (DC-centered)
    if save_qa:
        np.save(f"{qa_prefix}_kspace_os.npy", grid.astype(np.complex64))
        np.save(f"{qa_prefix}_kspace_mask.npy", (wgrid > 0).astype(np.float32))

    # IFFT -> image (oversampled), crop center to N^3
    #img_os = np.fft.ifftn(np.fft.ifftshift(grid))
    img_os = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(grid)))

    cx = Nx//2; cy = Ny//2; cz = Nz//2; h = N//2
    img = img_os[cx-h:cx-h+N, cy-h:cy-h+N, cz-h:cz-h+N]
    mag = np.abs(img); mag /= (mag.max() + eps)

    # PNG preview (central slice)
    z = N//2
    plt.figure(figsize=(6,6))
    plt.imshow(mag[:,:,z].T, cmap="gray", origin="lower")
    plt.axis("off")
    plt.title(f"Preview (CIC adjoint; slice {z})")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Saved {png_path}")

    # NIfTI (voxel size from FOV if provided)
    vox = (float(fov_mm)/float(N)) if (fov_mm is not None) else 1.0
    if fov_mm is None:
        print("[warn] FOV not provided; using voxel size 1.0 mm")
    affine = np.diag([vox, vox, vox, 1.0])
    nib.save(nib.Nifti1Image(mag.astype(np.float32), affine), nii_path)
    print(f"Saved {nii_path}")

    # PSFs for QA
    if save_qa:
        occ = (wgrid > 0).astype(np.float64)
        psf_sampling = np.abs(np.fft.ifftn(np.fft.ifftshift(occ)))
        psf_sampling /= (psf_sampling.max() + eps)
        psf_kernel = np.abs(np.fft.ifftn(np.fft.ifftshift(wgrid)))
        psf_kernel /= (psf_kernel.max() + eps)
        nib.save(nib.Nifti1Image(psf_sampling.astype(np.float32), affine), f"{qa_prefix}_psf_sampling.nii.gz")
        nib.save(nib.Nifti1Image(psf_kernel.astype(np.float32),   affine), f"{qa_prefix}_psf_kernel.nii.gz")
        print(f"Saved {qa_prefix}_psf_sampling.nii.gz and {qa_prefix}_psf_kernel.nii.gz")


# ------------------------------ Main ----------------------------------- #

def main():
    p = argparse.ArgumentParser(description="Generate/pack UTE3D radial NPZ and preview recon.")
    # Build parameters
    p.add_argument("--N", type=int, default=160, help="Matrix size (isotropic).")
    p.add_argument("--spokes", type=int, default=1200, help="Number of radial spokes.")
    p.add_argument("--readout", type=int, default=257, help="Readout samples per spoke (ODD ensures k=0 in symmetric mode).")
    p.add_argument("--kmax", type=float, default=0.5, help="Max k in cycles/FOV (e.g., 0.5 ~ Nyquist at edges).")
    p.add_argument("--center_out", action="store_true", help="Use center-out (0..kmax) trajectory.")
    p.add_argument("--pair_dirs", action="store_true", help="With center_out: also include negative directions (±d).")
    p.add_argument("--os_forward", type=float, default=2.0, help="Oversampling for preview (>=2 recommended).")
    p.add_argument("--T2star_ms", type=float, default=2.0, help="Approx T2* (ms) for physical simulator.")
    p.add_argument("--TE_us", type=float, default=80.0, help="Echo time (us) for physical simulator.")
    p.add_argument("--dwell_us", type=float, default=8.0, help="Dwell time (us) for physical simulator.")
    p.add_argument("--df_Hz", type=float, default=25.0, help="Off-resonance (Hz) for physical simulator.")
    p.add_argument("--noise_rel", type=float, default=0.0, help="Relative noise for synthetic kdata.")
    p.add_argument("--units", type=str, default="cycles/FOV", help="Trajectory units (informational).")
    p.add_argument("--fov_mm", type=float, default=None, help="Isotropic FOV (mm) for NIfTI voxel size.")
    p.add_argument("--use_physical", action="store_true", help="Use physical simulator (T2*, off-res) instead of static Gaussians.")
    # Optional input arrays
    p.add_argument("--kdata_npy", type=str, help="Path to existing kdata.npy (complex64).")
    p.add_argument("--ktraj_npy", type=str, help="Path to existing ktraj.npy (float32, shape [M,3]).")
    p.add_argument("--dcf_npy", type=str, help="Path to existing dcf.npy (float32). If omitted, computes safe DCF.")
    # QA options
    p.add_argument("--save_qa", action="store_true", help="Save oversampled k-space grid and PSF QA files.")
    p.add_argument("--qa_prefix", type=str, default="qa", help="Prefix for QA outputs.")
    # Output path
    p.add_argument("--outfile", type=str, required=True, help="Output NPZ file path.")
    # Reproducibility
    p.add_argument("--seed", type=int, default=1234, help="Seed for synthetic generation.")
    # Preview toggle
    p.add_argument("--no_preview", action="store_true", help="Skip PNG/NIfTI preview generation.")
    args = p.parse_args()

    spokes  = int(args.spokes)
    readout = int(args.readout)
    total   = spokes * readout

    # ---- Build or load ktraj ----
    if args.ktraj_npy:
        ktraj = np.load(args.ktraj_npy)
        if ktraj.dtype != np.float32: ktraj = ktraj.astype(np.float32)
        assert ktraj.ndim == 2 and ktraj.shape[1] == 3, "ktraj must be (M,3)"
    else:
        ktraj = gen_3d_radial_ktraj(spokes, readout, args.kmax,
                                    units=args.units,
                                    center_out=args.center_out,
                                    pair_dirs=args.pair_dirs,
                                    seed=args.seed)
    n_dirs_eff = ktraj.shape[0] // readout

    # ---- DC (k=0) sanity check ----
    has_dc = np.any(np.all(np.isclose(ktraj, 0.0, atol=1e-9), axis=1))
    print("has_dc:", bool(has_dc), "| center_out:", args.center_out, "| pair_dirs:", args.pair_dirs)

    # ---- Build or load kdata ----
    if args.kdata_npy:
        kdata = np.load(args.kdata_npy)
        if kdata.dtype != np.complex64: kdata = kdata.astype(np.complex64)
        assert kdata.ndim == 1, "kdata must be 1D (flattened samples)"
        if kdata.shape[0] != ktraj.shape[0]:
            raise ValueError(f"kdata length {kdata.shape[0]} != ktraj length {ktraj.shape[0]}")
    else:
        if args.use_physical:
            kdata = simulate_kdata_physical(ktraj, spokes, readout,
                                            T2star_ms=args.T2star_ms, TE_us=args.TE_us,
                                            dwell_us=args.dwell_us, df_Hz=args.df_Hz,
                                            seed=args.seed, noise_rel=args.noise_rel)
        else:
            kdata = simulate_kdata_gaussians_real(ktraj, seed=args.seed, noise_rel=args.noise_rel)

    # ---- Load or compute DCF ----
    if args.dcf_npy:
        dcf = np.load(args.dcf_npy)
        if dcf.dtype != np.float32: dcf = dcf.astype(np.float32)
        assert dcf.ndim == 1 and dcf.shape[0] == ktraj.shape[0], "dcf must be shape (M,)"
        m = dcf.mean()
        if m > 0: dcf = (dcf / m).astype(np.float32)
    else:
        dcf = compute_dcf_safe(ktraj, p=2.0, rmin=1e-3)

    # ---- Sanity checks ----
    if ktraj.shape[0] != total:
        print(f"[warn] ktraj length ({ktraj.shape[0]}) != spokes*readout ({total})")
    if kdata.shape[0] != ktraj.shape[0] or dcf.shape[0] != ktraj.shape[0]:
        raise ValueError("Size mismatch among kdata, ktraj, and dcf.")

    # ---- Build metadata ----
    meta = {
        "N": int(args.N),
        "spokes": spokes,
        "readout": readout,
        "kmax": float(args.kmax),
        "os_forward": float(args.os_forward),
        "T2star_ms": float(args.T2star_ms),
        "TE_us": float(args.TE_us),
        "dwell_us": float(args.dwell_us),
        "df_Hz": float(args.df_Hz),
        "noise_rel": float(args.noise_rel),
        "ktraj_units": f"{args.units} in [{'0,' if args.center_out else f'-{args.kmax}, '}{args.kmax}]",
        "dcf": "safe radial DCF (mean=1)",
        "sha1_kdata": sha1_of_array(kdata),
        "sha1_ktraj": sha1_of_array(ktraj),
        "sha1_dcf": sha1_of_array(dcf),
        "center_out": bool(args.center_out),
        "pair_dirs": bool(args.pair_dirs),
        "use_physical": bool(args.use_physical),
        "n_dirs_effective": int(n_dirs_eff),

    }

    # ---- Save NPZ ----
    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out), kdata=kdata.astype(np.complex64),
                       ktraj=ktraj.astype(np.float32),
                       dcf=dcf.astype(np.float32),
                       meta=meta)
    print_summary(kdata, ktraj, dcf, meta)
    print(f"\nSaved: {out.resolve()}")

    # ---- Preview recon (CIC gridding) ----
    if not args.no_preview:
        preview_recon_cic(kdata, ktraj, dcf,
                          N=args.N, osf=args.os_forward, fov_mm=args.fov_mm,
                          png_path="preview_slice.png",
                          nii_path="preview_recon.nii.gz",
                          save_qa=args.save_qa, qa_prefix=args.qa_prefix)


if __name__ == "__main__":
    main()
