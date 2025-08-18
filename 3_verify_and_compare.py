#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 16:40:42 2025

@author: alex
"""

import nibabel as nib, numpy as np, matplotlib.pyplot as plt

rec = nib.load("preview_recon.nii.gz").get_fdata()
ph  = nib.load("phantom_phys_mag.nii.gz").get_fdata()

# normalize
rec = rec / (rec.max()+1e-12)
ph  = ph  / (ph.max()+1e-12)

# metrics
mse  = np.mean((rec - ph)**2)
psnr = 10*np.log10(1.0/(mse + 1e-12))
ncc  = np.corrcoef(rec.ravel(), ph.ravel())[0,1]
print(f"PSNR={psnr:.2f} dB  NCC={ncc:.4f}")

# show central slices
z = rec.shape[2]//2
plt.figure(figsize=(12,5))
plt.subplot(1,3,1); plt.imshow(ph[:,:,z].T, cmap='gray', origin='lower'); plt.title("Phantom"); plt.axis('off')
plt.subplot(1,3,2); plt.imshow(rec[:,:,z].T, cmap='gray', origin='lower'); plt.title("Recon (CG)"); plt.axis('off')
plt.subplot(1,3,3); plt.imshow(np.abs(rec-ph)[:,:,z].T, cmap='gray', origin='lower'); plt.title("|Diff|"); plt.axis('off')
plt.tight_layout(); plt.show()
