# turner
3D radial project



python 1_make_npz.py --N 64 --spokes 3300 --readout 257 --center_out --pair_dirs --kmax 0.5 --noise_rel 0.0   --recon cg --lambda_l2 1e-4 --lambda_grad 5e-4 --max_iter 15 --fov_mm=20 --outfile ute3d_N64_S3300_cg.npz --save_qa --qa_prefix qa_N64_S3300
[info] pairing directions (±d) for center-out.


python recon_ute3d.py   --npz /Users/alex/AlexBadea_MyCodes/ute3d/ute3d_N64_S3300_cg.npz   --mode cg --backend finufft --osf 2.0   --lambda_l2 1e-4 --lambda_grad 5e-4 --max_iter 15   --fov_mm 20   --png N64_S3300_finufft_cg.png   --nii N64_S3300_finufft_cg.nii.gz
