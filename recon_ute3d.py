#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reconstruct UTE3D radial NPZ (kdata, ktraj, dcf, meta).

Backends
--------
- finufft (CPU): accurate NUFFT (recommended)
- torch (GPU optional): CIC-based adjoint/CG accelerated on CUDA if available
- torch_nufft (GPU/CPU): torchkbnufft (if installed)

Examples
--------
# CPU NUFFT (adjoint)
python recon_ute3d.py --npz ute3d_N64_S3300_cg.npz --mode adjoint --backend finufft --osf 2.0 --fov_mm 20

# CPU NUFFT + CG (Tikhonov + gradient-L2)
python recon_ute3d.py --npz ute3d_N96_S3k_cg.npz --mode cg --backend finufft \
  --lambda_l2 1e-3 --lambda_grad 1e-3 --max_iter 20 --fov_mm 20

# GPU CIC + CG (CUDA optional)
python recon_ute3d.py --npz ute3d_N64_S3300_cg.npz --mode cg --backend torch \
  --lambda_l2 1e-4 --lambda_grad 5e-4 --max_iter 15 --fov_mm 20

# Auto (prefers CUDA torch → finufft fallback)
python recon_ute3d.py --npz ute3d_N64_S3300_cg.npz --mode cg --backend auto --fov_mm 20

python recon_ute3d.py \
  --npz /Users/alex/AlexBadea_MyCodes/ute3d/ute3d_N64_S3300_cg.npz \
  --mode cg --backend finufft --osf 2.0 \
  --lambda_l2 1e-4 --lambda_grad 5e-4 --max_iter 15 \
  --fov_mm 20 \
  --png N64_S3300_finufft_cg.png \
  --nii N64_S3300_finufft_cg.nii.gz
  
"""
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# ---------------- Optional backends ----------------
_HAS_FINUFFT = False
try:
    import finufft
    _HAS_FINUFFT = True
except Exception:
    pass

_HAS_TORCH = False
_HAS_CUDA = False
try:
    import torch
    _HAS_TORCH = True
    _HAS_CUDA = torch.cuda.is_available()
except Exception:
    pass

_HAS_TORCHKB = False
try:
    from torchkbnufft import KbNufft, KbNufftAdjoint
    _HAS_TORCHKB = True
except Exception:
    pass

# ---------------- FINUFFT v1/v2–agnostic wrappers ----------------
# ---------------- FINUFFT v1/v2–agnostic wrappers (robust) ----------------
def _fnufft_type1(xj, yj, zj, cj, ms, mt, mu, isign=+1, eps=1e-9):
    """
    Try common Python bindings:
      v2: nufft3d1(xj,yj,zj,cj, ms,mt,mu, isign=, eps=)
      v1: nufft3d1(xj,yj,zj,cj, isign, eps, ms,mt,mu)
      tuple modes (old): nufft3d1(xj,yj,zj,cj, (ms,mt,mu), isign=, eps=)
      tuple modes (pos): nufft3d1(xj,yj,zj,cj, (ms,mt,mu), isign, eps)
    """
    import numpy as _np
    eps = float(eps)
    last_err = None
    try:
        return finufft.nufft3d1(xj, yj, zj, cj, ms, mt, mu, isign=isign, eps=eps)
    except TypeError as e:
        last_err = e
    try:
        return finufft.nufft3d1(xj, yj, zj, cj, isign, eps, ms, mt, mu)
    except TypeError as e:
        last_err = e
    try:
        return finufft.nufft3d1(xj, yj, zj, cj, (ms, mt, mu), isign=isign, eps=eps)
    except TypeError as e:
        last_err = e
    try:
        return finufft.nufft3d1(xj, yj, zj, cj, (ms, mt, mu), isign, eps)
    except TypeError as e:
        last_err = e
    raise TypeError(f"FINUFFT nufft3d1 signature not recognized: {last_err}")

def _fnufft_type2(xj, yj, zj, fk, isign=+1, eps=1e-9):
    """
    Try common Python bindings:
      v2: nufft3d2(xj,yj,zj, fk, isign=, eps=)
      v1: nufft3d2(xj,yj,zj, isign, eps, fk)
      (rare) kwonly eps/isign: nufft3d2(xj,yj,zj, fk, eps=, isign=)
    """
    eps = float(eps)
    last_err = None
    try:
        return finufft.nufft3d2(xj, yj, zj, fk, isign=isign, eps=eps)
    except TypeError as e:
        last_err = e
    try:
        return finufft.nufft3d2(xj, yj, zj, isign, eps, fk)
    except TypeError as e:
        last_err = e
    try:
        return finufft.nufft3d2(xj, yj, zj, fk, eps=eps, isign=isign)
    except TypeError as e:
        last_err = e
    raise TypeError(f"FINUFFT nufft3d2 signature not recognized: {last_err}")



# ---------------- Small helpers (NumPy) ----------------
def fft3c(x):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x)))

def ifft3c(X):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(X)))

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
    # cycles/FOV → radians; ensure contiguous float64 1D arrays
    u = np.ascontiguousarray(ktraj, dtype=np.float64) * (2*np.pi)
    return u[:, 0].copy(), u[:, 1].copy(), u[:, 2].copy()




# ---------------- Complex-safe Laplacians ----------------
def laplacian3d_np(x):
    """NumPy 6-neighbor Laplacian applied to complex x (component-wise)."""
    xr, xi = x.real, x.imag
    def L(u):
        up = np.pad(u, ((1,1),(1,1),(1,1)), mode='edge')
        return (up[2:,1:-1,1:-1] + up[:-2,1:-1,1:-1] +
                up[1:-1,2:,1:-1] + up[1:-1,:-2,1:-1] +
                up[1:-1,1:-1,2:] + up[1:-1,1:-1,:-2] - 6.0*u)
    return L(xr).astype(np.complex128) + 1j*L(xi).astype(np.complex128)

def laplacian3d_torch(x):
    """Torch 6-neighbor Laplacian applied to complex x (component-wise)."""
    import torch
    xr = torch.real(x); xi = torch.imag(x)
    def L(u):
        up = torch.nn.functional.pad(u, (1,1,1,1,1,1), mode='replicate')
        return (up[2:,1:-1,1:-1] + up[:-2,1:-1,1:-1] +
                up[1:-1,2:,1:-1] + up[1:-1,:-2,1:-1] +
                up[1:-1,1:-1,2:] + up[1:-1,1:-1,:-2] - 6.0*u)
    return torch.complex(L(xr), L(xi))

# ---------------- NumPy CG (for FINUFFT backend) ----------------
def cg_tikhonov_numpy(A, AH, y, lam_l2=0.0, lam_grad=0.0, N=64, max_iter=15, verbose=True):
    b = AH(y)
    x = np.zeros_like(b)
    def N_op(v):
        return AH(A(v)) + lam_l2*v + lam_grad*laplacian3d_np(v)
    r = b - N_op(x)
    p = r.copy()
    rs_old = np.vdot(r.ravel(), r.ravel())
    for it in range(1, max_iter+1):
        Ap = N_op(p)
        den = np.vdot(p.ravel(), Ap.ravel()) + 1e-30
        alpha = rs_old / den
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.vdot(r.ravel(), r.ravel())
        if verbose:
            print(f"[cg] it={it:02d} |r|={np.sqrt(rs_new.real):.3e}")
        if np.sqrt(rs_new.real) < 1e-6:
            break
        p = r + (rs_new/rs_old) * p
        rs_old = rs_new
    return x

# ---------------- Torch CG (for torch / torch_nufft backends) ----------------
def cg_tikhonov_torch(A, AH, y, lam_l2=0.0, lam_grad=0.0, N=64, max_iter=15, verbose=True):
    """
    Solve (A^H A + lam_l2 I + lam_grad L) x = A^H y via CG on torch tensors.
    A and AH take/return torch complex tensors.
    """
    import torch
    b = AH(y)
    x = torch.zeros_like(b)

    def N_op(v):
        return AH(A(v)) + lam_l2*v + lam_grad*laplacian3d_torch(v)

    r = b - N_op(x)
    p = r.clone()
    rs_old = torch.vdot(r.flatten(), r.flatten())  # complex scalar

    for it in range(1, max_iter+1):
        Ap = N_op(p)
        den = torch.vdot(p.flatten(), Ap.flatten())
        den = den + den.real.new_tensor(1e-30)  # epsilon with correct dtype/device
        alpha = rs_old / den

        x = x + alpha * p
        r = r - alpha * Ap

        rs_new = torch.vdot(r.flatten(), r.flatten())
        if verbose:
            print(f"[cg] it={it:02d} |r|={torch.sqrt(rs_new.real).item():.3e}")
        if torch.sqrt(rs_new.real).item() < 1e-6:
            break

        p = r + (rs_new/rs_old) * p
        rs_old = rs_new

    return x

# ---------------- FINUFFT operators (CPU) ----------------
def make_ops_finufft(N, osf, ktraj, dcf, eps=1e-9):
    if not _HAS_FINUFFT:
        raise RuntimeError("finufft not installed (pip install finufft)")
    N_os = int(round(osf * N))
    ms = mt = mu = N_os
    xj, yj, zj = ktraj_to_radians(ktraj)
    w = np.sqrt(np.maximum(dcf.astype(np.float64), 0.0))
    def A(x):
        Xos = fft3c(center_pad_to(x, (N_os, N_os, N_os))).astype(np.complex128, copy=False)
        fk = np.ascontiguousarray(Xos)                 # 3D, C-contiguous, complex128
        y  = _fnufft_type2(xj, yj, zj, fk, isign=+1, eps=eps)
        return (w * y).astype(np.complex128)

    def AH(y):
        grid = _fnufft_type1(xj, yj, zj, (w * y).astype(np.complex128),
                             ms, mt, mu, isign=+1, eps=eps)
        xos = ifft3c(grid.reshape(ms, mt, mu))
        return center_crop(xos, (N,N,N)).astype(np.complex128)
    return A, AH

# ---------------- Torch CIC operators (GPU/CPU) ----------------
def make_ops_torch_cic(N, osf, ktraj, dcf, device):
    import torch
    N_os = int(round(osf * N))
    # map cycles/FOV → grid coords [0..N_os-1], DC at center voxel
    u = torch.tensor(ktraj, dtype=torch.float64, device=device) * N_os + (N_os/2.0 - 0.5)
    w = torch.tensor(np.sqrt(np.maximum(dcf,0.0)), dtype=torch.float64, device=device)

    def cic_deposit_torch(y):
        Nx = Ny = Nz = N_os
        ux, uy, uz = u[:,0], u[:,1], u[:,2]
        i0x = torch.floor(ux).long().clamp(0, Nx-1); i1x = (i0x+1).clamp(0, Nx-1)
        i0y = torch.floor(uy).long().clamp(0, Ny-1); i1y = (i0y+1).clamp(0, Ny-1)
        i0z = torch.floor(uz).long().clamp(0, Nz-1); i1z = (i0z+1).clamp(0, Nz-1)
        wx = ux - torch.floor(ux); wy = uy - torch.floor(uy); wz = uz - torch.floor(uz)
        w000=(1-wx)*(1-wy)*(1-wz); w100=wx*(1-wy)*(1-wz)
        w010=(1-wx)*wy*(1-wz);     w110=wx*wy*(1-wz)
        w001=(1-wx)*(1-wy)*wz;     w101=wx*(1-wy)*wz
        w011=(1-wx)*wy*wz;         w111=wx*wy*wz

        grid = torch.zeros((Nx,Ny,Nz), dtype=y.dtype, device=device)
        def add(ix,iy,iz,wk): grid.index_put_((ix,iy,iz), y*wk, accumulate=True)
        add(i0x,i0y,i0z,w000); add(i1x,i0y,i0z,w100)
        add(i0x,i1y,i0z,w010); add(i1x,i1y,i0z,w110)
        add(i0x,i0y,i1z,w001); add(i1x,i0y,i1z,w101)
        add(i0x,i1y,i1z,w011); add(i1x,i1y,i1z,w111)

        occ = torch.zeros((Nx,Ny,Nz), dtype=torch.float64, device=device)
        def addw(ix,iy,iz,wk): occ.index_put_((ix,iy,iz), wk, accumulate=True)
        addw(i0x,i0y,i0z,w000); addw(i1x,i0y,i0z,w100)
        addw(i0x,i1y,i0z,w010); addw(i1x,i1y,i0z,w110)
        addw(i0x,i0y,i1z,w001); addw(i1x,i0y,i1z,w101)
        addw(i0x,i1y,i1z,w011); addw(i1x,i1y,i1z,w111)
        return grid / (occ.to(grid.dtype) + 1e-8)

    def cic_interpolate_torch(grid):
        Nx,Ny,Nz = grid.shape
        ux,uy,uz = u[:,0],u[:,1],u[:,2]
        i0x = torch.floor(ux).long().clamp(0, Nx-1); i1x = (i0x+1).clamp(0, Nx-1)
        i0y = torch.floor(uy).long().clamp(0, Ny-1); i1y = (i0y+1).clamp(0, Ny-1)
        i0z = torch.floor(uz).long().clamp(0, Nz-1); i1z = (i0z+1).clamp(0, Nz-1)
        wx = ux - torch.floor(ux); wy = uy - torch.floor(uy); wz = uz - torch.floor(uz)
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

    def A(x):
        import torch
        X = x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.complex128, device=device)
        Xos = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(X), dim=(0,1,2)))
        y  = cic_interpolate_torch(Xos)
        return (w * y).to(torch.complex128)

    def AH(y):
        import torch
        y = y if torch.is_tensor(y) else torch.tensor(y, dtype=torch.complex128, device=device)
        grid = cic_deposit_torch(w * y)
        xos = torch.fft.ifftshift(torch.fft.ifftn(torch.fft.ifftshift(grid), dim=(0,1,2)))
        xos = torch.fft.fftshift(xos)
        sx = (N_os - N)//2; sy = (N_os - N)//2; sz = (N_os - N)//2
        x = xos[sx:sx+N, sy:sy+N, sz:sz+N]
        return x.to(torch.complex128)

    return A, AH

# ---------------- Main CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="UTE3D recon from NPZ")
    ap.add_argument("--npz", required=True, help="Path to NPZ (kdata, ktraj, dcf, meta)")
    ap.add_argument("--mode", choices=["adjoint","cg"], default="adjoint")
    ap.add_argument("--backend", choices=["auto","finufft","torch","torch_nufft"], default="auto")
    ap.add_argument("--osf", type=float, default=2.0, help="Oversampling factor")
    ap.add_argument("--lambda_l2", type=float, default=1e-3)
    ap.add_argument("--lambda_grad", type=float, default=0.0)
    ap.add_argument("--max_iter", type=int, default=15)
    ap.add_argument("--fov_mm", type=float, default=None)
    ap.add_argument("--eps", type=float, default=1e-9)
    ap.add_argument("--png", default="recon_slice.png")
    ap.add_argument("--nii", default="recon.nii.gz")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    kdata = z["kdata"]; ktraj = z["ktraj"]; dcf = z["dcf"]; meta = z["meta"].item()
    N = int(meta.get("N", 128))

    print(f"Loaded: {args.npz}")
    print(f"N={N}, samples={kdata.size}, osf={args.osf}, backend={args.backend}")

    # pick backend
    backend = args.backend
    if backend == "auto":
        if _HAS_TORCH and _HAS_CUDA:
            backend = "torch"
        elif _HAS_FINUFFT:
            backend = "finufft"
        elif _HAS_TORCH:  # CPU torch if no CUDA
            backend = "torch"
        else:
            raise RuntimeError("No usable backend found. Install finufft or torch.")

    # ---- Recon paths ----
    if backend == "finufft":
        if not _HAS_FINUFFT:
            raise RuntimeError("FINUFFT not installed (pip install finufft).")
        if args.mode == "adjoint":
            N_os = int(round(args.osf * N))
            ms = mt = mu = N_os
            xj, yj, zj = ktraj_to_radians(ktraj)
            #cj = (kdata.astype(np.complex128) * dcf.astype(np.float64))
            #grid = _fnufft_type1(xj, yj, zj, cj, ms, mt, mu, isign=+1, eps=args.eps)
            
            
            cj = np.ascontiguousarray((kdata.astype(np.complex128) * dcf.astype(np.float64)))
            grid = _fnufft_type1(xj, yj, zj, cj, ms, mt, mu, isign=+1, eps=args.eps)


            img_os = ifft3c(grid.reshape(ms,mt,mu))
            img = center_crop(img_os, (N,N,N))
        else:
            A, AH = make_ops_finufft(N, args.osf, ktraj, dcf, eps=args.eps)
            img = cg_tikhonov_numpy(A, AH, kdata, lam_l2=args.lambda_l2,
                                    lam_grad=args.lambda_grad, N=N,
                                    max_iter=args.max_iter, verbose=True)

    elif backend == "torch":
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch not installed.")
        import torch
        device = "cuda" if _HAS_CUDA else "cpu"
        print(f"torch device: {device}")
        if args.mode == "adjoint":
            # Adjoint preview: deposit (kdata*DCF) to CIC grid → IFFT → crop
            N_os = int(round(args.osf * N))
            u = torch.tensor(ktraj, dtype=torch.float64, device=device) * N_os + (N_os/2.0 - 0.5)
            d = torch.tensor(kdata, dtype=torch.complex128, device=device)
            wfull = torch.tensor(dcf, dtype=torch.float64, device=device)
            # Deposit (inline)
            Nx = Ny = Nz = N_os
            ux, uy, uz = u[:,0], u[:,1], u[:,2]
            i0x = torch.floor(ux).long().clamp(0, Nx-1); i1x = (i0x+1).clamp(0, Nx-1)
            i0y = torch.floor(uy).long().clamp(0, Ny-1); i1y = (i0y+1).clamp(0, Ny-1)
            i0z = torch.floor(uz).long().clamp(0, Nz-1); i1z = (i0z+1).clamp(0, Nz-1)
            wx = ux - torch.floor(ux); wy = uy - torch.floor(uy); wz = uz - torch.floor(uz)
            w000=(1-wx)*(1-wy)*(1-wz); w100=wx*(1-wy)*(1-wz)
            w010=(1-wx)*wy*(1-wz);     w110=wx*wy*(1-wz)
            w001=(1-wx)*(1-wy)*wz;     w101=wx*(1-wy)*wz
            w011=(1-wx)*wy*wz;         w111=wx*wy*wz
            grid = torch.zeros((Nx,Ny,Nz), dtype=d.dtype, device=device)
            y_in = d * wfull.to(d.dtype)
            def add(ix,iy,iz,wk): grid.index_put_((ix,iy,iz), y_in*wk, accumulate=True)
            add(i0x,i0y,i0z,w000); add(i1x,i0y,i0z,w100)
            add(i0x,i1y,i0z,w010); add(i1x,i1y,i0z,w110)
            add(i0x,i0y,i1z,w001); add(i1x,i0y,i1z,w101)
            add(i0x,i1y,i1z,w011); add(i1x,i1y,i1z,w111)
            occ = torch.zeros((Nx,Ny,Nz), dtype=torch.float64, device=device)
            def addw(ix,iy,iz,wk): occ.index_put_((ix,iy,iz), wk, accumulate=True)
            addw(i0x,i0y,i0z,w000); addw(i1x,i0y,i0z,w100)
            addw(i0x,i1y,i0z,w010); addw(i1x,i1y,i0z,w110)
            addw(i0x,i0y,i1z,w001); addw(i1x,i0y,i1z,w101)
            addw(i0x,i1y,i1z,w011); addw(i1x,i1y,i1z,w111)
            grid = grid / (occ.to(grid.dtype) + 1e-8)
            xos = torch.fft.ifftshift(torch.fft.ifftn(torch.fft.ifftshift(grid), dim=(0,1,2)))
            xos = torch.fft.fftshift(xos)
            sx = (N_os - N)//2; sy=(N_os - N)//2; sz=(N_os - N)//2
            x = xos[sx:sx+N, sy:sy+N, sz:sz+N]
            img = x.detach().cpu().numpy()
        else:
            A, AH = make_ops_torch_cic(N, args.osf, ktraj, dcf, device=("cuda" if _HAS_CUDA else "cpu"))
            y = torch.tensor(kdata, dtype=torch.complex128, device=("cuda" if _HAS_CUDA else "cpu"))
            x = cg_tikhonov_torch(A, AH, y, lam_l2=args.lambda_l2,
                                  lam_grad=args.lambda_grad, N=N,
                                  max_iter=args.max_iter, verbose=True)
            img = x.detach().cpu().numpy()

    elif backend == "torch_nufft":
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch not installed.")
        if not _HAS_TORCHKB:
            raise RuntimeError("torchkbnufft not installed (pip install torchkbnufft).")
        import torch
        device = "cuda" if _HAS_CUDA else "cpu"
        print(f"torch_nufft device: {device}")
        om = torch.tensor(np.stack(ktraj_to_radians(ktraj), axis=0), dtype=torch.float32, device=device)
        im_size = torch.tensor([N, N, N])
        grid_size = torch.tensor([int(round(args.osf*N))]*3)
        kb = KbNufft(im_size=im_size, grid_size=grid_size).to(device)
        kbH = KbNufftAdjoint(im_size=im_size, grid_size=grid_size).to(device)
        d = torch.tensor(kdata, dtype=torch.complex64, device=device)
        w = torch.tensor(np.sqrt(np.maximum(dcf,0.0)), dtype=torch.float32, device=device)

        if args.mode == "adjoint":
            # approximate adjoint preview: push DCF into numerator
            x = kbH(d * (w**2), om=om)   # complex image
            img = x.detach().cpu().numpy()
        else:
            def A(v):
                return (w * kb(v, om=om)).to(torch.complex64)
            def AH(y):
                return kbH(w * y, om=om)
            y = d.to(torch.complex64)
            x = cg_tikhonov_torch(A, AH, y, lam_l2=args.lambda_l2,
                                  lam_grad=args.lambda_grad, N=N,
                                  max_iter=args.max_iter, verbose=True)
            img = x.detach().cpu().numpy()

    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Save PNG + NIfTI
    mag = np.abs(img); mag /= (mag.max() + 1e-12)
    zsl = N//2
    plt.figure(figsize=(6,6))
    plt.imshow(mag[:,:,zsl].T, cmap="gray", origin="lower")
    plt.axis("off"); plt.title(f"{backend} {args.mode} (slice {zsl})")
    plt.tight_layout(); plt.savefig(args.png, dpi=300); plt.close()

    vox = (float(args.fov_mm)/float(N)) if (args.fov_mm is not None) else 1.0
    if args.fov_mm is None:
        print("[warn] FOV not provided; using voxel size 1.0 mm")
    affine = np.diag([vox, vox, vox, 1.0])
    nib.save(nib.Nifti1Image(mag.astype(np.float32), affine), args.nii)
    print(f"Saved {args.png} and {args.nii}")

if __name__ == "__main__":
    main()
