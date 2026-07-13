"""v7 stage 1 (2 通道) 烟雾测试: 验证 forward+backward+AMP+Iterative+OAMFreqFilter 无 OOM/无维度错误"""
import sys
import torch
sys.path.insert(0, 'f:/d2nn')
import oam_crypt_d2nn as M
from oam_crypt_d2nn import (OAM_Crypt_D2NN, generate_rpp, encrypt_batch,
                            build_target_grid, calculate_center_psnr, security_ratio)
import time

device = torch.device("cuda")
print(f"Device: {device}")
print(f"CONFIG.layout = {M.CONFIG['layout']}")
print(f"CONFIG.curriculum = {M.CONFIG['curriculum']}")
print(f"Stage 1: {M.CONFIG['curriculum_stages'][0]}")

# 强制覆盖 quick_test_n 做最小烟雾测试
M.CONFIG["quick_test_n"] = 20  # 2 epoch × 20 batch = 40 iters

stage1 = M.CONFIG["curriculum_stages"][0]
t0 = time.time()
psnr, state, sr = M.train_one_stage(
    stage_cfg=stage1,
    stage_idx=0,
    n_stages=4,
    device=device,
    rpp_system=generate_rpp(M.CONFIG["size"], device),
    quick_test_n=20,
    num_layers=3,
    mid_ch=48,
    use_channel_attn=True,
    layout="oam_overlap",
)
elapsed = time.time() - t0
print(f"\n[SMOKE OK] stage 1 done in {elapsed:.0f}s")
print(f"  best PSNR_C = {psnr:.2f} dB")
print(f"  best SR_OAM = {sr:.4f}")
print(f"  state dict keys (n) = {len(state)}")
print(f"  has context_proj: {'context_proj.weight' in state}")
print(f"  has oam_filter mask: {'oam_filter.mask' in state}")
print(f"  has U-Net: {any('refine.' in k for k in state.keys())}")
print(f"  has D2NN layers: {any('layers.' in k for k in state.keys())}")
