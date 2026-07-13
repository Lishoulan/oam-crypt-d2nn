# -*- coding: utf-8 -*-
"""
v8 SecurityRatio 测试: Stage 1 2 通道 (PSNR_C 22-23 dB 才有意义)
- 合法解密 vs RPP 攻击 vs OAM 攻击
"""
import sys, os
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

def load_v8_model(ckpt_path, l_auth, z_list, slm_aware=False):
    # 用固定 generator 强制 rpp 与训练时一致 (训练时 manual_seed(42) 后 generate_rpp)
    rpp_gen = torch.Generator(device=device).manual_seed(42)
    rpp = generate_rpp(CONFIG['size'], device, generator=rpp_gen)
    model = OAM_Crypt_D2NN(
        size=CONFIG['size'], num_layers=CONFIG['num_layers'],
        wavelength=CONFIG['wavelength'], pixel_size=CONFIG['pixel_size'],
        z_layer=CONFIG['z_layer'], z0=CONFIG['z0'], rpp=rpp,
        oam_keys=l_auth, z_list=z_list,
        obj_encoding=CONFIG['obj_encoding'],
        theta_max=np.deg2rad(CONFIG['theta_max_deg']) if CONFIG.get('theta_max_deg') else None,
        slm_aware=slm_aware,
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

# Stage 1 配置
l_auth = [-25, 25]
z_list = [0.10, 0.55]
l_wrong = [-3, 3]  # 错误的 OAM 密钥
# 用相同 generator 强制 rpp_wrong 与 train_one_stage 内部生成的 rpp_wrong 一致
rpp_wrong_gen = torch.Generator(device=device).manual_seed(123)
rpp_wrong = generate_rpp(CONFIG['size'], device, generator=rpp_wrong_gen)
# 用训练时一致的 rpp (seed 42)
rpp_gen = torch.Generator(device=device).manual_seed(42)
rpp = generate_rpp(CONFIG['size'], device, generator=rpp_gen)

# 准备数据 (用 MNIST test 集前 200 个, 训练时评估用的就是它)
transform = transforms.Compose([transforms.ToTensor()])
full_test = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
mnist_sub = Subset(full_test, range(0, 200))  # 与训练时 test_loader 一致
img_size = CONFIG['size'] // 5
ds = MNISTQuadDataset(mnist_sub, img_size=img_size, num_channels=2)
loader = DataLoader(ds, batch_size=1, shuffle=False)

# 加载模型 (slm_aware=True 匹配训练时)
model = load_v8_model("oam_crypt_v8_stage1_best.pth", l_auth, z_list, slm_aware=True)

# 测试
results = {"auth": [], "rpp": [], "oam": []}
print(f"[v8 SECURITY] Stage 1 2 通道: 合法 / RPP 攻击 / OAM 攻击")
for i, batch_imgs in enumerate(loader):
    if i >= 30: break
    batch_imgs = batch_imgs.to(device)
    target = build_target_grid(batch_imgs, device, size=CONFIG['size'], layout='oam_overlap')

    # 合法
    cipher_auth = encrypt_batch(
        batch_imgs, l_auth, rpp,
        CONFIG['z0'], CONFIG['wavelength'], CONFIG['pixel_size'], device,
        size=CONFIG['size'], z_list=z_list,
        obj_encoding=CONFIG['obj_encoding'],
        theta_max=np.deg2rad(CONFIG['theta_max_deg']) if CONFIG.get('theta_max_deg') else None,
        layout='oam_overlap'
    )
    with torch.no_grad():
        pred_auth = model(cipher_auth)
    psnr_auth = calculate_center_psnr(pred_auth, target).item()
    results["auth"].append(psnr_auth)

    # RPP 攻击
    cipher_rpp = encrypt_batch(
        batch_imgs, l_auth, rpp_wrong,
        CONFIG['z0'], CONFIG['wavelength'], CONFIG['pixel_size'], device,
        size=CONFIG['size'], z_list=z_list,
        obj_encoding=CONFIG['obj_encoding'],
        theta_max=np.deg2rad(CONFIG['theta_max_deg']) if CONFIG.get('theta_max_deg') else None,
        layout='oam_overlap'
    )
    with torch.no_grad():
        pred_rpp = model(cipher_rpp)
    psnr_rpp = calculate_center_psnr(pred_rpp, target).item()
    results["rpp"].append(psnr_rpp)

    # OAM 攻击
    cipher_oam = encrypt_batch(
        batch_imgs, l_wrong, rpp,
        CONFIG['z0'], CONFIG['wavelength'], CONFIG['pixel_size'], device,
        size=CONFIG['size'], z_list=z_list,
        obj_encoding=CONFIG['obj_encoding'],
        theta_max=np.deg2rad(CONFIG['theta_max_deg']) if CONFIG.get('theta_max_deg') else None,
        layout='oam_overlap'
    )
    with torch.no_grad():
        pred_oam = model(cipher_oam)
    psnr_oam = calculate_center_psnr(pred_oam, target).item()
    results["oam"].append(psnr_oam)

    print(f"  样本 {i+1}: 合法 {psnr_auth:.2f} dB  RPP攻击 {psnr_rpp:.2f} dB  OAM攻击 {psnr_oam:.2f} dB")

# 统计
print(f"\n{'='*60}")
print(f"[v8 STAGE 1 SECURITY SUMMARY]")
print(f"{'='*60}")
auth_mean = np.mean(results["auth"])
rpp_mean = np.mean(results["rpp"])
oam_mean = np.mean(results["oam"])
print(f"合法解密:     {auth_mean:.2f} dB")
print(f"RPP 攻击:    {rpp_mean:.2f} dB  (差 {auth_mean-rpp_mean:+.2f} dB)")
print(f"OAM 攻击:    {oam_mean:.2f} dB  (差 {auth_mean-oam_mean:+.2f} dB)")
sr_rpp = rpp_mean / auth_mean
sr_oam = oam_mean / auth_mean
print(f"\nSR_RPP = {sr_rpp:.4f}  (目标 < 0.3)")
print(f"SR_OAM = {sr_oam:.4f}  (目标 < 0.3)")
if sr_rpp < 0.3 and sr_oam < 0.3:
    print(f"✅ 工程级加密达成: 攻击后接近噪声, 不可识别原图")
else:
    print(f"⚠️  安全性未完全达成: 需要更强攻击或更长密钥")
