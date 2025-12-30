import subprocess

# ── Configuration ──────────────────────────────────────────────────────────────
max_mask_num = 26   # change this to your desired maximum mask number
pipeline_script = "ReSeg3D.py"
# ────────────────────────────────────────────────────────────────────────────────

# every odd mask from 0 to max_mask_num
mask_num_list1 = [1, 3, 12, 18, 19, 22, 26, 20, 0, 4, 6, 10, 15, 7]
mask_num_list2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


for mask_num in mask_num_list1:
    print(f"▶ Running mask {mask_num}/{max_mask_num}")
    subprocess.run([
        "python",
        pipeline_script,
        "--mask", str(mask_num)
    ], check=True)
