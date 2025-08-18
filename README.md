# turner
3D radial project



#python 1_make_npz.py --N 64 --spokes 3300 --readout 257 --center_out --pair_dirs --kmax 0.5 --noise_rel 0.0   --recon cg --lambda_l2 1e-4 --lambda_grad 5e-4 --max_iter 15 --fov_mm=20 --outfile ute3d_N64_S3300_cg.npz --save_qa --qa_prefix qa_N64_S3300
#[info] pairing directions (±d) for center-out.

python 2_make_npz.py   --sim physical   --N 64 --spokes 3300 --readout 257   --center_out --pair_dirs --kmax 0.5   --T2star_ms 2.0 --TE_us 80 --dwell_us 8 --df_Hz 25   --noise_rel 0.0 --fov_mm 20   --outfile ute3d_N64_S3300_phys.npz   --save_phantom --phantom_prefix phantom_phys   --recon cg --lambda_l2 1e-4 --lambda_grad 5e-4 --max_iter 15

python recon_ute3d.py   --npz /Users/alex/AlexBadea_MyCodes/ute3d/ute3d_N64_S3300_cg.npz   --mode cg --backend finufft --osf 2.0   --lambda_l2 1e-4 --lambda_grad 5e-4 --max_iter 15   --fov_mm 20   --png N64_S3300_finufft_cg.png   --nii N64_S3300_finufft_cg.nii.gz


python recon_ute3d.py \
  --npz /Users/alex/AlexBadea_MyCodes/ute3d/ute3d_N64_S3300_phys.npz \
  --mode cg --backend finufft --osf 2.0 \
  --lambda_l2 1e-4 --lambda_grad 5e-4 --max_iter 20 \
  --fov_mm 20 \
  --png N64_S3300_finufft_cg.png \
  --nii N64_S3300_finufft_cg.nii.gz

  
Tuning tips
Slight streaks → raise --lambda_grad (e.g., 1e-3).
Over-smoothing → lower --lambda_grad (e.g., 2e-4).
If SNR is too high/clean (since noise_rel=0), add a touch of noise (e.g., --noise_rel 0.01) to test robustness.


python 5_residuals_plus.py   --npz /Users/alex/AlexBadea_MyCodes/ute3d/ute3d_N64_S3300_phys.npz   --complex_npz recon_complex.npz   --osf 2.0 --prefix res_phys_complex

