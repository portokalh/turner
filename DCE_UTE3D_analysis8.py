#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DCE UTE3D / FLASH analysis with baseline strategy, masking, ROI curves, QA ROI masks,
and --auto_aif N selection of early-peaking, high-PE voxels.

DCE UTE3D analysis (semi-quantitative) — UTE3D/DCE helper

Defaults (tweak with flags):

--auto_aif_ttp_max 4 → only voxels with very early TTP (≤4 frames)

--auto_aif_pe_pct 95 → keep top 5% PE_max within the mask

--auto_aif_min_dist_vox 3 → enforce spacing between AIF voxels

--auto_aif_radius_vox 1 → r=1 voxel ROIs (ideal for QA)

Now with your specifics:
- Default baseline frames: 0 1 2  (injection at ~3:00 → safely pre-bolus)
- Default frame spacing: 89.0 seconds (1 min 29 s)
- Optional --flip_angle to record 5° or 11° in metadata
- Optional --injection_time_sec to mark injection in plot & metadata

Outputs
- PE_max.nii.gz, AUC.nii.gz, TTP_index.nii.gz, slope_in.nii.gz, slope_out.nii.gz, S0.nii.gz
- global_curve.csv (frame, time_sec, time_min, percent_enhancement)
- global_curve.png (with dashed line at injection time if provided)
- run_meta.json (parsed method info + arguments used)


  
  
  
  
  
%run "/Users/alex/AlexBadea_MyExperiments/DennisTurner/Code/DCE_UTE3D_analysis8.py" \
  --nifti "/Users/alex/AlexBadea_MyExperiments/DennisTurner/Data/100925/21_1_UTE3D_UTE3D_builtin_traj.nii.gz" \
  --method "/Users/alex/AlexBadea_MyExperiments/DennisTurner/Data/100925/21_1_UTE3D_UTE3D_builtin_traj.method" \
  --force_ras 1 \
  --baseline_strategy mean --baseline_frames 0 1 2 \
  --time_spacing 89 --flip_angle 5 --injection_time_sec 180 \
  --auto_mask 1 --auto_mask_pct 97 --smooth_win 3 \
  --roi_mm "CSF1:-0.3,1.16,0.31,0.4" \
  --roi_mm "CSF2:1.88,1.28,0.21,0.4" \
  --roi_mm "V1:2.59,-3.63,-1.33,0.4" \
  --roi_mm "V2:-0.09,-3.58,-0.68,0.4" \
  --roi_mm "T1:-1.97,0.21,1.32,0.4" \
  --roi_mm "T2:3.58,-0.03,1.33,0.4" \
  --auto_aif 2 --auto_aif_ttp_max 5 --auto_aif_pe_pct 99.5 \
  --auto_aif_min_dist_vox 6 --auto_aif_radius_vox 1 \
  --normalize_by_roi CSF2 \
  --outdir "/Users/alex/AlexBadea_MyExperiments/DennisTurner/Data/100925/ute3d_mm_outputs_normed"
  
  
 %run "/Users/alex/AlexBadea_MyExperiments/DennisTurner/Code/DCE_UTE3D_analysis8.py" \
  --nifti "/Users/alex/AlexBadea_MyExperiments/DennisTurner/Data/100925/21_1_UTE3D_UTE3D_builtin_traj.nii.gz" \
  --method "/Users/alex/AlexBadea_MyExperiments/DennisTurner/Data/100925/21_1_UTE3D_UTE3D_builtin_traj.method" \
  --force_ras 1 \
  --baseline_strategy mean --baseline_frames 0 1 2 \
  --time_spacing 89 --flip_angle 5 --injection_time_sec 180 \
  --auto_mask 1 --auto_mask_pct 97 --smooth_win 3 \
  --normalize_by_roi CSF2 \
  --roi_mm "CSF1:-0.3,1.16,0.31,0.4" \
  --roi_mm "CSF2:1.88,1.28,0.21,0.4" \
  --roi_mm "V1:2.59,-3.63,-1.33,0.4" \
  --roi_mm "V2:-0.09,-3.58,-0.68,0.4" \
  --roi_mm "T1:-1.97,0.21,1.32,0.4" \
  --roi_mm "T2:3.58,-0.03,1.33,0.4" \
  --roi_mm "AIF1:-0.37,-5.10,-1.80,0.2" \
  --roi_mm "AIF2:2.80,-4.90,1.80,0.2" \
  --auto_aif 0 \
  --outdir "/Users/alex/AlexBadea_MyExperiments/DennisTurner/Data/100925/ute3d_mm_outputs_AB"


%run "/Users/alex/AlexBadea_MyExperiments/DennisTurner/Code/DCE_UTE3D_analysis8.py" \
 --nifti "/Users/alex/AlexBadea_MyExperiments/DennisTurner/Data/100925/2_1_DCE_FLASH.nii.gz" \
  --method "/Users/alex/AlexBadea_MyExperiments/DennisTurner/Data/100925/2_1_DCE_FLASH.method" \
  --resample_to "/Users/alex/AlexBadea_MyExperiments/DennisTurner/Data/100925/21_1_UTE3D_UTE3D_builtin_traj.nii.gz" \
  --force_ras 1 \
  --baseline_strategy first --baseline_frames 0 \
  --time_spacing 285 --flip_angle 12 --injection_time_sec -1200 \
  --auto_mask 1 --auto_mask_pct 97 --smooth_win 3 \
  --roi_mm "CSF1:-0.3,1.16,0.31,0.4" \
  --roi_mm "CSF2:1.88,1.28,0.21,0.4" \
  --roi_mm "V1:2.59,-3.63,-1.33,0.4" \
  --roi_mm "V2:-0.09,-3.58,-0.68,0.4" \
  --roi_mm "T1:-1.97,0.21,1.32,0.4" \
  --roi_mm "T2:3.58,-0.03,1.33,0.4" \
  --destripe 1 --destripe_axis 0 --destripe_zthr 6 \
  --roi_mm "AIF1:-0.37,-5.10,-1.80,0.4" \
  --roi_mm "AIF2:2.80,-4.90,1.80,0.4" \
  --auto_aif 0 \
  --normalize_by_roi CSF2 \
  --outdir "/Users/alex/AlexBadea_MyExperiments/DennisTurner/Data/100925/dce_flash_mm_outputs_AB"



"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DCE analysis (UTE3D / FLASH) with millimeter ROIs, optional resampling & destriping.
Saves BOTH unnormalized and, if requested, ROI-normalized curves/plots/tables.

Outputs (always)
- PE_max.nii.gz, AUC.nii.gz, TTP_index.nii.gz, slope_in.nii.gz, slope_out.nii.gz, S0.nii.gz
- global_curve.png, global_curve.csv
- roi_curves.csv, roi_pe_curves.png, roi_raw_curves.png
- roi_summary.csv  (PE_max, TTP, slopes per ROI)

If --normalize_by_roi <ROI> is given, also saves:
- roi_curves_norm.csv, roi_pe_curves_norm.png, roi_raw_curves_norm.png
- roi_summary_norm.csv

Author: AlexB + ChatGPT
"""

import argparse, json, csv, re
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# -------------------------- Helpers: I/O & geometry --------------------------

def parse_method(path):
    meta = {}
    if not path: return meta
    p = Path(path)
    if not p.exists(): return meta
    try: txt = p.read_text(errors='ignore')
    except Exception: return meta
    def grab(pats, cast=float):
        for pat in pats:
            m = re.search(pat, txt)
            if m:
                try: return cast(m.group(1))
                except Exception: return None
        return None
    meta['TR_ms'] = grab([r"PVM_RepetitionTime\s*=\s*([\d\.Ee\+\-]+)"])
    meta['TE_ms'] = grab([r"PVM_EchoTime\s*=\s*([\d\.Ee\+\-]+)"])
    meta['FlipAngle_deg'] = grab([r"PVM_FlipAngle\s*=\s*([\d\.Ee\+\-]+)"])
    meta['Dynamics'] = grab([r"PVM_NRepetitions\s*=\s*(\d+)"], cast=int)
    return meta

def img_voxsize_mm(img):
    z = img.header.get_zooms()
    return (float(z[0]), float(z[1]), float(z[2]))

def mm_to_vox(affine, xyz_mm):
    mm = np.array([xyz_mm[0], xyz_mm[1], xyz_mm[2], 1.0], dtype=np.float64)
    vij = np.linalg.inv(affine) @ mm
    return vij[:3]

def vox_to_mm(affine, ijk):
    v = np.array([ijk[0], ijk[1], ijk[2], 1.0], dtype=np.float64)
    mm = affine @ v
    return mm[:3]

def resample_4d_to_like(mov_img, ref_img, order=1):
    """Resample 4D moving image onto reference image grid with linear interp."""
    from scipy.ndimage import affine_transform
    mov = mov_img.get_fdata(dtype=np.float32)
    ref = ref_img
    out_shape = ref.shape[:3] + (mov.shape[3],)
    A = np.linalg.inv(mov_img.affine) @ ref.affine   # ref-vox -> mov-vox
    M = A[:3,:3]; b = A[:3,3]
    out = np.empty(out_shape, dtype=np.float32)
    for t in range(mov.shape[3]):
        out[..., t] = affine_transform(
            mov[..., t], matrix=M, offset=b,
            output_shape=ref.shape[:3],
            order=order, mode='nearest', cval=0.0, prefilter=(order>1)
        )
    return nib.Nifti1Image(out, ref.affine, ref.header)

def force_ras(img):
    """Return a copy reoriented to RAS+. Keeps data 4D. Works with old nibabel."""
    import nibabel as nib
    # Fast path (newer nibabel)
    ac = getattr(nib, "as_closest_canonical", None)
    if ac is not None:
        return ac(img)

    # Fallback (older nibabel) via orientations
    from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation, inv_ornt_aff
    cur = io_orientation(img.affine)
    tgt = axcodes2ornt(('R','A','S'))
    xfm = ornt_transform(cur, tgt)
    data = apply_orientation(img.get_fdata(dtype=np.float32), xfm)
    new_aff = img.affine @ inv_ornt_aff(xfm, img.shape)
    return nib.Nifti1Image(data, new_aff, img.header)


# -------------------------- Artifact mitigation (QA) --------------------------

def destripe_image_volume(vol4d, axis=0, zscore_thr=6.0):
    """
    Crude despiking for a persistent bright stripe in magnitude images.
    Replaces outlier columns/rows (persistent across T) with the median of neighbors.
    """
    v = vol4d.copy()
    X, Y, Z, T = v.shape
    if axis == 0:
        prof = np.nanmean(v**2, axis=(2,3)).mean(axis=1)  # (X,)
    else:
        prof = np.nanmean(v**2, axis=(2,3)).mean(axis=0)  # (Y,)
    mu, sd = np.mean(prof), np.std(prof) + 1e-6
    outliers = np.where((prof - mu)/sd > zscore_thr)[0]
    if outliers.size == 0:
        return v, []
    for idx in outliers:
        if axis == 0:
            left = max(0, idx-1); right = min(X-1, idx+1)
            repl = np.median(v[[left, right], :, :, :], axis=0)
            v[idx, :, :, :] = repl
        else:
            top = max(0, idx-1); bot = min(Y-1, idx+1)
            repl = np.median(v[:, [top, bot], :, :], axis=1)
            v[:, idx, :, :] = repl
    return v, outliers.tolist()

# -------------------------- Core DCE metrics --------------------------

def compute_metrics(data4d, baseline_idx, time_spacing=1.0, baseline_strategy='mean'):
    assert data4d.ndim == 4, f'Expected 4D NIfTI (x,y,z,t), got {data4d.shape}'
    T = data4d.shape[-1]
    baseline_idx = np.array(sorted(set(int(i) for i in baseline_idx if 0 <= int(i) < T)))
    if baseline_idx.size == 0:
        raise ValueError('No valid baseline frame indices.')
    if baseline_strategy == 'first':
        S0 = data4d[..., 0]
    else:
        S0 = np.mean(data4d[..., baseline_idx], axis=-1)
    S0_safe = np.where(S0 == 0, np.finfo(np.float32).eps, S0)
    PE = 100.0 * (data4d - S0_safe[..., None]) / S0_safe[..., None]
    PE_max = np.max(PE, axis=-1)
    ttp_idx = np.argmax(PE, axis=-1)
    try:
        auc = np.trapezoid(PE, dx=time_spacing, axis=-1)
    except Exception:
        auc = np.trapz(PE, dx=time_spacing, axis=-1)  # fallback (deprecated)
    base_end = baseline_idx.max(); last_idx = PE.shape[-1]-1
    t1 = np.maximum(base_end+1, np.minimum(ttp_idx, last_idx))
    dy_in = np.take_along_axis(PE, ttp_idx[...,None], axis=-1)[...,0] - PE[..., base_end]
    dt_in = (t1 - base_end) * time_spacing
    dt_in_safe = np.where(dt_in == 0, np.finfo(np.float32).eps, dt_in)
    slope_in = dy_in / dt_in_safe
    dy_out = PE[..., last_idx] - np.take_along_axis(PE, ttp_idx[...,None], axis=-1)[...,0]
    dt_out = (last_idx - ttp_idx) * time_spacing
    dt_out_safe = np.where(dt_out == 0, np.finfo(np.float32).eps, dt_out)
    slope_out = dy_out / dt_out_safe
    return dict(PE=PE, S0=S0, PE_max=PE_max, TTP_idx=ttp_idx, AUC=auc,
                slope_in=slope_in, slope_out=slope_out)

def save_nifti_like(ref_img, data3d, outpath):
    nib.save(nib.Nifti1Image(data3d.astype(np.float32), ref_img.affine, ref_img.header), str(outpath))

# -------------------------- ROIs in mm (primary) --------------------------

def sphere_mask_mm(shape_xyz, center_mm, r_mm, affine):
    """
    Boolean mask (X,Y,Z) for a sphere in *mm* space.
    Uses full affine to honor orientation (RAS/LPS/etc).
    """
    X, Y, Z = shape_xyz
    xs = np.arange(X, dtype=np.float32)
    ys = np.arange(Y, dtype=np.float32)
    zs = np.arange(Z, dtype=np.float32)
    I, J, K = np.meshgrid(xs, ys, zs, indexing='ij')  # (X,Y,Z)
    ijk = np.stack([I, J, K], axis=-1)  # (X,Y,Z,3)
    # map to mm: mm = ijk @ A[:3,:3]^T + A[:3,3]
    A = affine.astype(np.float32)
    mm = ijk @ A[:3,:3].T + A[:3,3]
    d2 = ((mm[...,0]-center_mm[0])**2 +
          (mm[...,1]-center_mm[1])**2 +
          (mm[...,2]-center_mm[2])**2)
    return d2 <= (r_mm**2)

def extract_roi_curves_mm(img, data4d, baseline_idx, rois_mm):
    """
    rois_mm: list of {name, center_mm(x,y,z), r_mm}
    Returns dict[name]->dict(raw, pe, nvox, center_vox, r_mm, r_vox_est)
    """
    T = data4d.shape[-1]
    affine = img.affine
    zooms = img_voxsize_mm(img)
    g = (zooms[0]*zooms[1]*zooms[2])**(1/3)
    S0 = np.mean(data4d[..., baseline_idx], axis=-1)
    S0_safe = np.where(S0 == 0, np.finfo(np.float32).eps, S0)

    out = {}
    for roi in rois_mm:
        name = roi['name']; ctr = roi['center_mm']; rmm = float(roi['r_mm'])
        mask = sphere_mask_mm(data4d.shape[:3], ctr, rmm, affine)
        vox = data4d[mask, :]
        nvox = int(vox.shape[0])
        center_vox = tuple(np.round(mm_to_vox(affine, ctr)).astype(int).tolist())
        if nvox == 0:
            raw = np.full(T, np.nan); pe = np.full(T, np.nan)
            r_vox_est = np.nan
        else:
            raw = vox.mean(axis=0)
            S0_roi = S0_safe[mask].mean()
            pe = 100.0*(raw - S0_roi)/S0_roi
            r_vox_est = rmm / g
        out[name] = dict(raw=raw, pe=pe, nvox=nvox, r_mm=rmm, r_vox=r_vox_est,
                         center_mm=list(ctr), center_vox=center_vox)
    return out

def moving_avg(y, w):
    if w is None or w < 3 or w % 2 == 0: return y
    k = w//2; pad = np.pad(y, (k,k), mode='edge'); ker = np.ones(w)/w
    return np.convolve(pad, ker, mode='valid')

# -------------------------- Auto-AIF (voxel domain) --------------------------

def auto_aif_pick(M, mask, N, ttp_max, pe_pct, min_dist_vox, radius_vox):
    if not (N and N > 0): return []
    tt = M["TTP_idx"].astype(int)
    pe_max = M["PE_max"].astype(float)
    T = int(M["PE"].shape[-1])
    ttp_max = int(min(max(0, ttp_max), T-1))
    early = (tt <= ttp_max)
    if not np.any(mask & early): return []

    pe_vals = pe_max[mask & early]
    if pe_vals.size == 0: return []
    thr = np.percentile(pe_vals, float(pe_pct))
    cand = np.array(np.where(mask & early & (pe_max >= thr))).T
    if cand.size == 0: return []

    # sort by PE_max
    scores = pe_max[cand[:,0], cand[:,1], cand[:,2]]
    cand = cand[np.argsort(scores)[::-1]]

    # washout requirement
    slope_out = M["slope_out"]; PE = M["PE"]
    last_q_start = max(1, int(0.75 * T))
    good = []
    for c in cand:
        x,y,z = map(int, c)
        if slope_out[x,y,z] >= 0:
            continue
        peak = float(np.max(PE[x,y,z,:]))
        tail = float(np.mean(PE[x,y,z,last_q_start:]))
        if (peak - tail) < 1.0:
            continue
        good.append(c)
    if len(good) == 0: return []

    # spacing
    selected = []
    mind2 = max(1, int(min_dist_vox))**2
    for c in good:
        if len(selected) >= int(N): break
        if all(np.sum((c - s)**2) >= mind2 for s in selected):
            selected.append(c)

    return [dict(x=int(c[0]), y=int(c[1]), z=int(c[2]), r_vox=int(max(1, radius_vox))) for c in selected]

# -------------------------- CSV / plot writers --------------------------

def write_roi_curves_csv(path, R, time_spacing):
    T = len(next(iter(R.values()))['raw'])
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        header = ['frame','time_sec','time_min']
        for rn in R: header += [f'{rn}_raw', f'{rn}_PE']
        w.writerow(header)
        for i in range(T):
            t = i*float(time_spacing)
            row = [i, t, t/60.0]
            for rn in R:
                row += [float(R[rn]['raw'][i]), float(R[rn]['pe'][i])]
            w.writerow(row)

def write_roi_summary_csv(path, R, baseline_frames, time_spacing):
    T = len(next(iter(R.values()))['pe'])
    times_min = np.arange(T)*float(time_spacing)/60.0
    base_end = int(np.max(baseline_frames))
    last_idx = T - 1

    def sdiv(a, b):
        b = float(b)
        return float(a)/(b if b != 0 else np.finfo(np.float32).eps)

    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['roi','nvox','PE_max','TTP_index','TTP_min',
                    'slope_in_%permin','slope_out_%permin'])
        for rn, rr in R.items():
            pe = np.asarray(rr['pe'], dtype=float)
            if not np.all(np.isfinite(pe)) or pe.size == 0:
                w.writerow([rn, int(rr.get('nvox',0))] + [float('nan')]*6)
                continue
            ttp_idx = int(np.nanargmax(pe))
            pe_max = float(pe[ttp_idx])
            ttp_min = float(times_min[ttp_idx])

            t1 = max(base_end + 1, min(ttp_idx, last_idx))
            dy_in = pe[ttp_idx] - pe[base_end]
            dt_in_min = (t1 - base_end) * (float(time_spacing)/60.0)
            slope_in = sdiv(dy_in, dt_in_min)

            dy_out = pe[last_idx] - pe[ttp_idx]
            dt_out_min = (last_idx - ttp_idx) * (float(time_spacing)/60.0)
            slope_out = sdiv(dy_out, dt_out_min)

            w.writerow([rn, int(rr.get('nvox',0)), pe_max, ttp_idx, ttp_min,
                        slope_in, slope_out])

def plot_roi_set(R, times, out_png, inj_sec, smooth_win, mode='pe', title=''):
    plt.figure(figsize=(7.8,4.4))
    for rn in R:
        y = R[rn]['pe'] if mode == 'pe' else R[rn]['raw']
        plt.plot(times/60.0, moving_avg(y, smooth_win), label=rn)
    if inj_sec is not None:
        plt.axvline(float(inj_sec)/60.0, linestyle='--')
    plt.xlabel('Time (min)')
    plt.ylabel('% enhancement' if mode=='pe' else 'Raw signal (a.u.)')
    plt.title(title)
    plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    
    
    
def aif_sanity_report(R, baseline_frames, time_spacing, out_csv, aif_prefix="AIF", early_frac=0.25,
                      min_drop_pct=1.0, verbose=True):
    """
    For ROIs whose names start with `aif_prefix`, compute:
      - TTP_index, TTP_min
      - PE_max
      - slope_out_%permin  (should be negative)
    Also checks a simple 'washout' criterion: peak - late_tail >= min_drop_pct (%).

    Saves a CSV and returns a list of dict rows. Prints a one-liner verdict per AIF if verbose.
    """
    rows = []
    if not R:
        return rows

    # timebase
    T = len(next(iter(R.values()))["pe"])
    times_min = (np.arange(T) * float(time_spacing)) / 60.0
    base_end = int(np.max(baseline_frames))
    last_idx = T - 1
    last_q_start = max(1, int((1.0 - early_frac) * T))  # default last 25% of frames

    def sdiv(a, b):
        b = float(b)
        return float(a) / (b if b != 0 else np.finfo(np.float32).eps)

    # collect AIF-like names
    aif_names = [rn for rn in R if rn.upper().startswith(aif_prefix.upper())]

    for rn in aif_names:
        rr = R[rn]
        pe = np.asarray(rr["pe"], dtype=float)
        if not np.all(np.isfinite(pe)) or pe.size == 0:
            rows.append(dict(roi=rn, note="non-finite curve",
                             TTP_index=np.nan, TTP_min=np.nan,
                             PE_max=np.nan, slope_out_pct_per_min=np.nan,
                             washout_ok=False, verdict="FAIL"))
            continue

        # PE_max and TTP
        ttp_idx = int(np.nanargmax(pe))
        ttp_min = float(times_min[ttp_idx])
        pe_max = float(pe[ttp_idx])

        # slope_out: from TTP to last, in %/min
        dy_out = pe[last_idx] - pe[ttp_idx]
        dt_out_min = (last_idx - ttp_idx) * (float(time_spacing) / 60.0)
        slope_out = sdiv(dy_out, dt_out_min)

        # washout: drop from peak to mean tail
        tail = float(np.nanmean(pe[last_q_start:]))
        drop = pe_max - tail
        washout_ok = bool(drop >= float(min_drop_pct))

        # verdict
        verdict_bits = []
        verdict_bits.append("early" if ttp_idx == np.nanargmin([np.nanargmax(R[x]['pe']) if np.all(np.isfinite(R[x]['pe'])) else np.inf for x in aif_names]) else "not-earliest")
        verdict_bits.append("washout" if washout_ok else "no-washout")
        verdict_bits.append("neg-slope" if slope_out < 0 else "pos-slope")
        verdict = "OK" if (washout_ok and slope_out < 0) else "CHECK"

        row = dict(
            roi=rn,
            TTP_index=ttp_idx,
            TTP_min=ttp_min,
            PE_max=pe_max,
            slope_out_pct_per_min=slope_out,
            washout_ok=washout_ok,
            verdict=verdict
        )
        rows.append(row)

        if verbose:
            print(f"[AIF sanity] {rn}: TTP={ttp_min:.2f} min "
                  f"PE_max={pe_max:.2f}% slope_out={slope_out:.3f} %/min "
                  f"{'washout✓' if washout_ok else 'washout✗'} → {verdict}")

    # write CSV
    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    return rows
    

# -------------------------- Main --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nifti', required=True)
    ap.add_argument('--method')

    # timing / baseline
    ap.add_argument('--baseline_frames', nargs='+', type=int, default=[0,1,2])
    ap.add_argument('--baseline_strategy', choices=['mean','first'], default='mean')
    ap.add_argument('--time_spacing', type=float, default=89.0)
    ap.add_argument('--flip_angle', type=float, default=5.0)
    ap.add_argument('--injection_time_sec', type=float, default=180.0)
    ap.add_argument('--flip_time', type=int, default=0)

    # orientation / resampling & destriping
    ap.add_argument('--force_ras', type=int, default=0, help='Reorient to RAS+ before processing (0/1).')
    ap.add_argument('--resample_to', default=None, help='Reference NIfTI to resample onto (optional).')
    ap.add_argument('--destripe', type=int, default=0, help='Apply QA destriping (0/1).')
    ap.add_argument('--destripe_axis', type=int, default=0, help='0=X vertical stripe, 1=Y horizontal stripe.')
    ap.add_argument('--destripe_zthr', type=float, default=6.0)

    # masks & smoothing
    ap.add_argument('--auto_mask', type=int, default=1)
    ap.add_argument('--auto_mask_pct', type=float, default=97.0)
    ap.add_argument('--mask', default=None)
    ap.add_argument('--smooth_win', type=int, default=3)
    ap.add_argument('--export_pe_series', type=int, default=0)

    # ROIs (mm-first design)
    ap.add_argument('--roi_mm', action='append', default=[], help='name:x_mm,y_mm,z_mm,r_mm')
    ap.add_argument('--roi_mm_csv', default=None, help='CSV with columns name,x_mm,y_mm,z_mm,r_mm')
    ap.add_argument('--save_roi_masks', type=int, default=1)
    ap.add_argument('--roi_labelmap_name', default='roi_labelmap.nii.gz')

    # Auto AIF (voxel domain)
    ap.add_argument('--auto_aif', type=int, default=0, help='How many auto AIFs to add (0=off).')
    ap.add_argument('--auto_aif_ttp_max', type=int, default=3)
    ap.add_argument('--auto_aif_pe_pct', type=float, default=99.0)
    ap.add_argument('--auto_aif_min_dist_vox', type=int, default=6)
    ap.add_argument('--auto_aif_radius_vox', type=int, default=1)

    # Optional ROI normalization (second set of outputs)
    ap.add_argument('--normalize_by_roi', default=None,
                    help='Name of ROI to ratio-normalize others (e.g., CSF2).')

    ap.add_argument('--outdir', default='dce_mm_outputs')
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # load
    img0 = nib.load(args.nifti)
    img = img0
    if args.force_ras:
        img = force_ras(img)
    if args.resample_to:
        ref_img = nib.load(args.resample_to)
        if args.force_ras:
            ref_img = force_ras(ref_img)
        img = resample_4d_to_like(img, ref_img, order=1)

    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 4: raise RuntimeError(f'Expected 4D NIfTI; got {data.shape}')
    if args.flip_time: data = data[..., ::-1]

    # optional destripe
    if args.destripe:
        v, idx = destripe_image_volume(data, axis=int(args.destripe_axis), zscore_thr=float(args.destripe_zthr))
        if len(idx) == 0:  # try other axis
            v2, idx2 = destripe_image_volume(data, axis=1-int(args.destripe_axis), zscore_thr=float(args.destripe_zthr))
            if len(idx2) > 0:
                data = v2
        else:
            data = v

    # metrics
    M = compute_metrics(data,
                        baseline_idx=args.baseline_frames,
                        time_spacing=args.time_spacing,
                        baseline_strategy=args.baseline_strategy)

    # save maps
    save_nifti_like(img, M['PE_max'], outdir/'PE_max.nii.gz')
    save_nifti_like(img, M['AUC'], outdir/'AUC.nii.gz')
    save_nifti_like(img, M['TTP_idx'].astype(np.float32), outdir/'TTP_index.nii.gz')
    save_nifti_like(img, M['slope_in'], outdir/'slope_in.nii.gz')
    save_nifti_like(img, M['slope_out'], outdir/'slope_out.nii.gz')
    save_nifti_like(img, M['S0'], outdir/'S0.nii.gz')
    if args.export_pe_series:
        nib.save(nib.Nifti1Image(M['PE'].astype(np.float32), img.affine, img.header),
                 str(outdir/'PE_series.nii.gz'))

    # mask
    S0_map = M['S0'].astype(np.float32)
    if args.mask:
        mask = nib.load(args.mask).get_fdata().astype(bool)
    else:
        if args.auto_mask:
            pos = S0_map[S0_map>0]
            thr = np.percentile(pos, float(args.auto_mask_pct)) if pos.size else 0.0
            mask = S0_map >= thr
        else:
            mask = np.ones_like(S0_map, dtype=bool)

    # global curve
    PE = M['PE']; masked_pe = PE[mask,:]; masked_raw = data[mask,:]
    global_pe = moving_avg(masked_pe.mean(axis=0), args.smooth_win)
    global_raw = moving_avg(masked_raw.mean(axis=0), args.smooth_win)

    with (outdir/'global_curve.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame','time_sec','time_min','raw_mean_masked','percent_enhancement_mean_masked'])
        for i,(r,p) in enumerate(zip(global_raw, global_pe)):
            t = i*args.time_spacing; w.writerow([i,t,t/60.0,float(r),float(p)])

    times = np.arange(global_pe.size)*args.time_spacing
    plt.figure(figsize=(7.4,4.3)); plt.plot(times/60.0, global_pe)
    if args.injection_time_sec is not None: plt.axvline(args.injection_time_sec/60.0, linestyle='--')
    plt.xlabel('Time (min)'); plt.ylabel('% enhancement (masked mean)'); plt.title('Global Mean DCE Curve')
    plt.tight_layout(); plt.savefig(outdir/'global_curve.png', dpi=200); plt.close()

    # ---------- build ROI list in mm ----------
    rois_mm = []
    for item in args.roi_mm:
        try:
            name, rest = item.split(':',1)
            x,y,z,r = [float(v) for v in rest.split(',')]
            rois_mm.append({'name':name, 'center_mm':(x,y,z), 'r_mm':float(r)})
        except Exception:
            pass
    if args.roi_mm_csv and Path(args.roi_mm_csv).exists():
        with Path(args.roi_mm_csv).open() as f:
            rdr = csv.reader(f); _ = next(rdr, None)
            for r in rdr:
                if len(r) < 5: continue
                name, x, y, z, rmm = r[:5]
                rois_mm.append({'name':name, 'center_mm':(float(x),float(y),float(z)), 'r_mm':float(rmm)})
    # dedupe by name (first occurrence wins)
    seen, rois_mm_final = set(), []
    for r in rois_mm:
        if r['name'] in seen: continue
        seen.add(r['name']); rois_mm_final.append(r)

    # ---------- Auto AIF in voxel space, then convert to mm and append ----------
    if int(args.auto_aif) > 0:
        picks = auto_aif_pick(M, mask,
                              N=int(args.auto_aif),
                              ttp_max=int(args.auto_aif_ttp_max),
                              pe_pct=float(args.auto_aif_pe_pct),
                              min_dist_vox=int(args.auto_aif_min_dist_vox),
                              radius_vox=int(args.auto_aif_radius_vox))
        g = (np.prod(img_voxsize_mm(img)))**(1/3)
        for i, p in enumerate(picks, start=1):
            name = f"AIF{i}"
            if any(r['name'] == name for r in rois_mm_final):
                name = f"AIF{i}_auto"
            ctr_mm = tuple(vox_to_mm(img.affine, (p['x'], p['y'], p['z'])))
            rois_mm_final.append({'name':name, 'center_mm':ctr_mm, 'r_mm':float(max(g*0.9, 0.5))})

    # ---------- ROI extraction ----------
    R = {}
    if rois_mm_final:
        R = extract_roi_curves_mm(img, data, baseline_idx=args.baseline_frames, rois_mm=rois_mm_final)

        # optional save ROI masks + labelmap
        if args.save_roi_masks:
            label = np.zeros(data.shape[:3], dtype=np.int16)
            saved = []
            for idx, r in enumerate(rois_mm_final, start=1):
                mask_r = sphere_mask_mm(data.shape[:3], r['center_mm'], r['r_mm'], img.affine)
                nib.save(nib.Nifti1Image(mask_r.astype(np.uint8), img.affine, img.header),
                         str(outdir / f"roi_{r['name']}.nii.gz"))
                label[mask_r] = idx
                saved.append({
                    'index': idx,
                    'name': r['name'],
                    'center_mm': list(r['center_mm']),
                    'r_mm': float(r['r_mm']),
                    'center_vox': R[r['name']]['center_vox'],
                    'r_vox_est': float(R[r['name']]['r_vox']),
                })
            nib.save(nib.Nifti1Image(label, img.affine, img.header), str(outdir/args.roi_labelmap_name))
            with (outdir/'roi_set_mm.json').open('w') as f: f.write(json.dumps(saved, indent=2))

        # ---------- UNNORMALIZED outputs ----------
        write_roi_curves_csv(outdir/'roi_curves.csv', R, args.time_spacing)
        write_roi_summary_csv(outdir/'roi_summary.csv', R, args.baseline_frames, args.time_spacing)
        plot_roi_set(R, times, outdir/'roi_pe_curves.png', args.injection_time_sec, args.smooth_win,
                     mode='pe', title='ROI % Enhancement')
        plot_roi_set(R, times, outdir/'roi_raw_curves.png', args.injection_time_sec, args.smooth_win,
                     mode='raw', title='ROI Raw Signal')
        
       

        # NEW: AIF sanity (unnormalized)
        aif_sanity_report(
            R,
            baseline_frames=args.baseline_frames,
            time_spacing=args.time_spacing,
            out_csv=outdir/'aif_sanity.csv',
            aif_prefix='AIF',        # names like AIF1, AIF2 ...
            early_frac=0.25,         # last 25% for tail
            min_drop_pct=1.0,        # require ≥1% drop peak→tail
            verbose=True
        )


        # ---------- NORMALIZED set (optional) ----------
        norm_name = args.normalize_by_roi
        if norm_name and norm_name in R and np.all(np.isfinite(R[norm_name]['raw'])):
            Rn = {}
            ref_raw = R[norm_name]['raw'].astype(float).copy()
            ref_b = np.mean(ref_raw[np.asarray(args.baseline_frames, int)])
            ref_raw /= (ref_b if ref_b != 0 else 1.0)
            for rn, rr in R.items():
                raw_n = rr['raw'].astype(float) / ref_raw
                b = np.mean(raw_n[np.asarray(args.baseline_frames, int)])
                pe_n = 100.0 * (raw_n - b) / (b if b != 0 else np.finfo(np.float32).eps)
                Rn[rn] = dict(raw=raw_n, pe=pe_n, nvox=rr.get('nvox',0))

            write_roi_curves_csv(outdir/'roi_curves_norm.csv', Rn, args.time_spacing)
            write_roi_summary_csv(outdir/'roi_summary_norm.csv', Rn, args.baseline_frames, args.time_spacing)
            plot_roi_set(Rn, times, outdir/'roi_pe_curves_norm.png', args.injection_time_sec, args.smooth_win,
                         mode='pe', title='ROI % Enhancement (normalized)')
            plot_roi_set(Rn, times, outdir/'roi_raw_curves_norm.png', args.injection_time_sec, args.smooth_win,
                         mode='raw', title=f'ROI Raw (normalized by {norm_name})')

    # meta
    meta = parse_method(args.method) if args.method else {}
    meta['FlipAngle_deg'] = float(args.flip_angle)
    meta_out = {
        'method_meta': meta, 'nifti_shape': list(data.shape),
        'baseline_frames': [int(v) for v in args.baseline_frames], 'baseline_strategy': args.baseline_strategy,
        'time_spacing_sec': float(args.time_spacing), 'injection_time_sec': float(args.injection_time_sec),
        'flip_time': int(args.flip_time),
        'force_ras': int(args.force_ras),
        'resampled_to': str(args.resample_to) if args.resample_to else None,
        'destripe': int(args.destripe), 'destripe_axis': int(args.destripe_axis),
        'auto_mask': int(args.auto_mask), 'auto_mask_pct': float(args.auto_mask_pct),
        'smooth_win': int(args.smooth_win),
        'roi_mm_count': int(len(rois_mm_final)),
        'auto_aif': int(args.auto_aif), 'auto_aif_ttp_max': int(args.auto_aif_ttp_max),
        'auto_aif_pe_pct': float(args.auto_aif_pe_pct), 'auto_aif_min_dist_vox': int(args.auto_aif_min_dist_vox),
        'auto_aif_radius_vox': int(args.auto_aif_radius_vox),
        'normalize_by_roi': args.normalize_by_roi
    }
    (outdir/'run_meta.json').write_text(json.dumps(meta_out, indent=2))

if __name__ == '__main__':
    main()
