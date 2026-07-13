# -*- coding: utf-8 -*-
"""
v8 SLM 加载测试: 对比 Stage 1, 2, 4 三个模型的 SLM 加载损耗
数字加载: model(slm_aware=False) - 关闭 SLM 感知
SLM 加载: model(slm_aware=True) - 开启 8-bit 量化感知
"""
import sys, os, time
import numpy as np
import torch
sys.path.insert(0, '.')
from oam_crypt_d2nn import (
    CONFIG, OAM_Crypt_D2NN, generate_rpp, encrypt_batch, build_target_grid,
    calculate_center_psnr, MNISTQuadDataset
)
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

device = torch.device(CONFIG['device'])
torch.manual_seed(42)

def load_v8_model(ckpt_path, l_auth, z_list, slm_aware):
    rpp = generate_rpp(CONFIG['size'], device)
    model = OAM_Crypt_D2NN(
        size=CONFIG['size'], num_layers=CONFIG['num_layers'],
        wavelength=CONFIG['wavelength'], pixel_size=CONFIG['pixel_size'],
        z_layer=CONFIG['z_layer'], z0=CONFIG['z0'], rpp=rpp,
        oam_keys=l_auth, z_list=z_list,
        obj_encoding=CONFIG['obj_encoding'],
        theta_max=np.deg2rad(CONFIG['theta_max_deg']) if CONFIG.get('theta_max_deg') else None,
        slm_aware=slm_aware,  # 关键: 切换
        use_channel_attn=True, mid_ch=CONFIG['mid_ch'],
        iterative_refine=False,
        oam_freq_filter=True,
        use_polar_conv=True,
        polar_n_r=32, polar_n_theta=96,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model

# 准备数据
transform = transforms.Compose([transforms.ToTensor()])
full_train = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
mnist_sub = Subset(full_train, range(50))
img_size = CONFIG['size'] // 5
ds = MNISTQuadDataset(mnist_sub, img_size=img_size, num_channels=10)
loader = DataLoader(ds, batch_size=1, shuffle=False)

# 三个 stage 配置
configs = [
    ("Stage 1 (2 通道)",  "oam_crypt_v8_stage1_best.pth", [-25, 25],            [0.10, 0.55]),
    ("Stage 2 (5 通道)",  "oam_crypt_v8_stage2_best.pth", [-25, -15, 0, 15, 25], [0.10, 0.20, 0.35, 0.45, 0.55]),
    ("Stage 4 (10 通道)", "oam_crypt_v8_stage4_best.pth", [-25, -20, -15, -10, -5, 5, 10, 15, 20, 25],
     [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]),
]

results = []
for name, ckpt, l_auth, z_list in configs:
    if not os.path.exists(ckpt):
        print(f"  跳过 {name}: {ckpt} 不存在")
        continue
    print(f"\n[SLM LOAD] {name}")
    rpp = generate_rpp(CONFIG['size'], device)
    n_ch = len(l_auth)

    # 加载两个模型: 数字版 (slm_aware=False) + SLM 版 (slm_aware=True)
    model_digital = load_v8_model(ckpt, l_auth, z_list, slm_aware=False)
    model_slm = load_v8_model(ckpt, l_auth, z_list, slm_aware=True)

    psnr_digital, psnr_slm = 0.0, 0.0
    n_samples = 0
    for batch_imgs in loader:
        if n_samples >= 10: break
        batch_imgs = batch_imgs.to(device)
        batch_imgs_n = batch_imgs[:, :n_ch]
        target = build_target_grid(batch_imgs_n, device, size=CONFIG['size'], layout='oam_overlap')
        cipher = encrypt_batch(
            batch_imgs_n, l_auth, rpp,
            CONFIG['z0'], CONFIG['wavelength'], CONFIG['pixel_size'], device,
            size=CONFIG['size'], z_list=z_list,
            obj_encoding=CONFIG['obj_encoding'],
            theta_max=np.deg2rad(CONFIG['theta_max_deg']) if CONFIG.get('theta_max_deg') else None,
            layout='oam_overlap'
        )
        with torch.no_grad():
            pred_digital = model_digital(cipher)
            psnr_d = calculate_center_psnr(pred_digital, target).item()
            pred_slm = model_slm(cipher)
            psnr_s = calculate_center_psnr(pred_slm, target).item()
        psnr_digital += psnr_d
        psnr_slm += psnr_s
        n_samples += 1
    psnr_digital /= n_samples
    psnr_slm /= n_samples
    loss = psnr_digital - psnr_slm
    print(f"  数字 {psnr_digital:.2f} dB → SLM {psnr_slm:.2f} dB  损耗 {loss:+.2f} dB")
    results.append((name, psnr_digital, psnr_slm, loss))

print(f"\n{'='*60}")
print(f"[v8 SLM 加载总结]")
print(f"{'='*60}")
print(f"{'配置':<20}{'数字':>8}{'SLM':>8}{'损耗':>10}")
for name, d, s, l in results:
    print(f"{name:<20}{d:>8.2f}{s:>8.2f}{l:>+10.2f}")
