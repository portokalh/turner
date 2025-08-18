#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 16:45:58 2025

@author: alex
python 4_compare_recons.py   --nufft N64_S3300_finufft_cg.nii.gz   --preview preview_recon.nii.gz   --phantom phantom_phys_mag.nii.gz   --prefix cmp_phys
python 4_compare_recons.py   --nufft N64_S3300_finufft_cg.nii.gz   --preview preview_recon.nii.gz   --phantom phantom_phys_mag.nii.gz   --prefix cmp_phys  
python 4_compare_recons.py   --nufft N64_S3300_finufft_cg.nii.gz   --preview N64_S3300_finufft_adj.nii.gz   --prefix cmp_finufft_ref

"""

#!/usr/bin/env python3
import argparse, numpy as np, nibabel as nib, matplotlib.pyplot as plt
from pathlib import Path

def norm01(x):
    x = x - x.min()
    m = x.max()
    return x / m if m > 0 else x

def center_match(a, target_shape):
    A = np.array(a)
    out = np.zeros(target_shape, dtype=A.dtype)
    src = []; dst = []
    for i in range(3):
        ai, ti = A.shape[i], target_shape[i]
        if ai >= ti:
            s = (ai - ti)//2
            src.append(slice(s, s+ti)); dst.append(slice(0, ti))
        else:
            s = (ti - ai)//2
            src.append(slice(0, ai));   dst.append(slice(s, s+ai))
    out[dst[0], dst[1], dst[2]] = A[src[0], src[1], src[2]]
    return out

def save_slice(vol, title, out_png):
    z = vol.shape[2]//2
    plt.figure(figsize=(6,6))
    plt.imshow(vol[:,:,z].T, cmap="gray", origin="lower")
    plt.axis("off"); plt.title(title); plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nufft", required=True, help="NUFFT-CG NIfTI (e.g., N64_S3300_finufft_cg.nii.gz)")
    ap.add_argument("--preview", required=True, help="Preview CG NIfTI (e.g., preview_recon.nii.gz)")
    ap.add_argument("--phantom", help="(optional) Phantom/ref NIfTI, for extra metrics")
    ap.add_argument("--prefix", default="cmp", help="Output prefix for PNGs/metrics")
    args = ap.parse_args()

    fin = nib.load(args.nufft).get_fdata().astype(np.float64)
    pre = nib.load(args.preview).get_fdata().astype(np.float64)
    if pre.shape != fin.shape:
        pre = center_match(pre, fin.shape)

    fin_n = norm01(fin); pre_n = norm01(pre)
    mse  = float(np.mean((fin_n - pre_n)**2))
    psnr = float(10*np.log10(1.0/(mse + 1e-12)))
    ncc  = float(np.corrcoef(fin_n.ravel(), pre_n.ravel())[0,1])

    save_slice(fin_n, "NUFFT-CG (image space)", f"{args.prefix}_nufft.png")
    save_slice(pre_n, "Preview CG (image space)", f"{args.prefix}_preview.png")
    diff = np.abs(fin_n - pre_n)
    diff_n = diff / (diff.max() + 1e-12)
    save_slice(diff_n, "Absolute difference (scaled)", f"{args.prefix}_diff.png")

    lines = [f"MSE={mse:.6e}", f"PSNR={psnr:.2f} dB", f"NCC={ncc:.4f}"]
    if args.phantom:
        ph = nib.load(args.phantom).get_fdata().astype(np.float64)
        if ph.shape != fin.shape:
            ph = center_match(ph, fin.shape)
        ph_n = norm01(ph)
        mse_nufft  = float(np.mean((fin_n - ph_n)**2))
        mse_preview= float(np.mean((pre_n - ph_n)**2))
        psnr_nufft = float(10*np.log10(1.0/(mse_nufft + 1e-12)))
        psnr_prev  = float(10*np.log10(1.0/(mse_preview + 1e-12)))
        ncc_nufft  = float(np.corrcoef(fin_n.ravel(), ph_n.ravel())[0,1])
        ncc_prev   = float(np.corrcoef(pre_n.ravel(), ph_n.ravel())[0,1])
        lines += [
            "--- vs Phantom ---",
            f"NUFFT-CG:   PSNR={psnr_nufft:.2f} dB  NCC={ncc_nufft:.4f}",
            f"Preview CG: PSNR={psnr_prev:.2f} dB   NCC={ncc_prev:.4f}",
        ]

    txt = "\n".join(lines)
    print(txt)
    Path(f"{args.prefix}_metrics.txt").write_text(txt)

if __name__ == "__main__":
    main()
