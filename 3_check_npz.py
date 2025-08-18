import numpy as np
import matplotlib.pyplot as plt

# --- Load your NPZ ---
npz_path = "ute3d_radial.npz"   # <- change if needed
data = np.load(npz_path, allow_pickle=True)
kdata = data["kdata"]
ktraj = data["ktraj"]
dcf   = data["dcf"]
meta  = data["meta"].item()

print(f"kdata: {kdata.shape} {kdata.dtype}")
print(f"ktraj: {ktraj.shape} {ktraj.dtype}  min/max: {ktraj.min():.6f}/{ktraj.max():.6f}")
print(f"dcf  : {dcf.shape} {dcf.dtype}  mean: {dcf.mean():.6f}  min/max: {dcf.min():.6f}/{dcf.max():.6f}")
print("meta :", meta)

# --- Utility: radial norm |k| ---
kr = np.linalg.norm(ktraj.astype(np.float64), axis=1)

# --- 3D scatter of trajectory (downsample for speed/clarity) ---
N = ktraj.shape[0]
step = max(N // 5000, 1)    # target ~5k points
xyz = ktraj[::step]
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], s=1, alpha=0.5)
ax.set_title("3D Trajectory (downsampled)")
ax.set_xlabel("kx (cycles/FOV)")
ax.set_ylabel("ky (cycles/FOV)")
ax.set_zlabel("kz (cycles/FOV)")
plt.tight_layout()
plt.savefig("trajectory_3d.png", dpi=300)
plt.show()

# --- DCF histogram ---
plt.figure(figsize=(6,4))
plt.hist(dcf, bins=80)
plt.title("DCF Histogram")
plt.xlabel("dcf value")
plt.ylabel("count")
plt.tight_layout()
plt.savefig("dcf_histogram.png", dpi=300)
plt.show()

# --- Radial profile: dcf vs |k| ---
bins = np.linspace(kr.min(), kr.max(), 60)
idx  = np.digitize(kr, bins)
med  = np.array([np.median(dcf[idx==b]) if np.any(idx==b) else np.nan for b in range(1, len(bins)+1)])
centers = (bins[:-1] + bins[1:]) / 2

plt.figure(figsize=(6,4))
plt.plot(centers, med[:-1], lw=2)
plt.title("Radial DCF Profile (median per |k| bin)")
plt.xlabel("|k| (cycles/FOV)")
plt.ylabel("median dcf")
plt.tight_layout()
plt.savefig("dcf_radial_profile.png", dpi=300)
plt.show()

# --- Optional quick checks ---
print(f"|k| mean: {kr.mean():.4f}, |k| min/max: {kr.min():.4f}/{kr.max():.4f}")
r = np.corrcoef(dcf, kr**2)[0,1]
print(f"corr(dcf, |k|^2) = {r:.4f}")
