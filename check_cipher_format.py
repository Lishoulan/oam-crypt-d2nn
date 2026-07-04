# -*- coding: utf-8 -*-
"""检查 U_cipher 的数值格式与 SLM 兼容性"""
import torch
import numpy as np
from oam_crypt_d2nn import (
    CONFIG, generate_rpp, OAM_Crypt_D2NN,
    MNISTQuadDataset, encrypt_batch
)
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

device = torch.device(CONFIG["device"])
torch.manual_seed(42)
np.random.seed(42)

# 数据
transform = transforms.Compose([transforms.ToTensor()])
full_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
mnist_test = Subset(full_test, range(32))
test_dataset = MNISTQuadDataset(mnist_test)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# RPP (固定种子, 与训练一致)
rpp_system = generate_rpp(CONFIG["size"], device)

# 取一批明文
batch_imgs = next(iter(test_loader)).to(device)

# 加密
U_cipher = encrypt_batch(
    batch_imgs, CONFIG["l_auth"], rpp_system,
    CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
    z_list=CONFIG["z_list"]
)

print("=" * 60)
print("U_cipher 格式分析")
print("=" * 60)
print(f"数据类型:           {U_cipher.dtype}")
print(f"形状 (张量):        {tuple(U_cipher.shape)}")
print(f"分辨率:             {U_cipher.shape[-1]} x {U_cipher.shape[-2]} 像素")
print(f"物理尺寸:           {U_cipher.shape[-1] * CONFIG['pixel_size'] * 1e3:.2f} mm x "
      f"{U_cipher.shape[-2] * CONFIG['pixel_size'] * 1e3:.2f} mm")
print()
print("复振幅统计:")
amp = torch.abs(U_cipher)
phase = torch.angle(U_cipher)
print(f"  振幅 |U|   范围: [{amp.min().item():.4f}, {amp.max().item():.4f}], "
      f"均值={amp.mean().item():.4f}, std={amp.std().item():.4f}")
print(f"  相位 angle 范围: [{phase.min().item():.4f}, {phase.max().item():.4f}] rad")
print(f"  相位 angle 范围: [{phase.min().item()/np.pi:.3f}π, {phase.max().item()/np.pi:.3f}π]")

# 单张全息图的强度统计 (SLM 通常加载的灰度图)
I_cipher = (amp ** 2).cpu().numpy()
print()
print(f"强度图 I=|U|^2 统计 (单张样本):")
print(f"  范围: [{I_cipher.min():.6f}, {I_cipher.max():.6f}]")
print(f"  均值: {I_cipher.mean():.6f}, std: {I_cipher.std():.6f}")
print(f"  动态范围 (max/min): {I_cipher.max() / (I_cipher.min() + 1e-12):.1f}")

# 归一化到 8-bit 灰度 (SLM 典型输入)
I_norm = I_cipher[0] / I_cipher[0].max()
I_8bit = (I_norm * 255).astype(np.uint8)
print()
print(f"归一化到 8-bit 灰度后 (SLM 加载格式):")
print(f"  范围: [0, 255] (uint8)")
print(f"  均值: {I_8bit.mean():.2f}")
print(f"  std:  {I_8bit.std():.2f}")

# 纯相位 SLM 加载所需的相位图
phase_np = phase[0].cpu().numpy()
phase_8bit = ((phase_np + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
print()
print(f"纯相位 SLM 加载 (相位映射到 0-255 灰度):")
print(f"  原始相位范围: [{phase_np.min():.4f}, {phase_np.max():.4f}] rad")
print(f"  映射后范围:   [{phase_8bit.min()}, {phase_8bit.max()}] (uint8)")
print(f"  映射公式:     gray = round((phase + π) / (2π) × 255)")

print()
print("=" * 60)
print("SLM 兼容性诊断")
print("=" * 60)
print(f"当前代码分辨率: {CONFIG['size']} × {CONFIG['size']}")
print(f"当前像素尺寸:   {CONFIG['pixel_size'] * 1e6:.1f} μm")
print(f"波长:           {CONFIG['wavelength'] * 1e9:.0f} nm")
print()
print("典型商用 SLM 参数对照:")
slm_specs = [
    ("Holoeye LET-5019",      1920, 1080, 6.4),
    ("Holoeye PLUTO-2.1",    1920, 1080, 8.0),
    ("Meadowlark HSP1920",   1920, 1152, 7.6),
    ("Meadowlark D5123",      512,  512, 15.0),
    ("Thorlabs EXULUS-4K1",  3840, 2160, 3.5),
    ("Hamamatsu X13138",     1272,  800, 12.5),
]
print(f"  {'型号':<25} {'分辨率':<12} {'像素(μm)':<10} {'是否兼容':<10}")
for name, w, h, ps in slm_specs:
    size_ok = (w >= CONFIG['size']) and (h >= CONFIG['size'])
    pix_ok = abs(ps - CONFIG['pixel_size']*1e6) < 2.0  # 2 μm 容差
    status = "可加载" if (size_ok and pix_ok) else ("需缩放" if size_ok else "分辨率不足")
    print(f"  {name:<25} {w}x{h:<8} {ps:<10.1f} {status}")

print()
print("关键诊断:")
print(f"  1. 分辨率 {CONFIG['size']}x{CONFIG['size']} -> SLM 需至少 {CONFIG['size']}x{CONFIG['size']}")
print(f"     (推荐 zero-pad 到 SLM 原生分辨率, 居中放置)")
print(f"  2. 像素尺寸 {CONFIG['pixel_size']*1e6:.1f} μm -> 需匹配 SLM 像素尺寸")
print(f"     (不匹配会导致传播距离 z0={CONFIG['z0']*1e2:.1f} cm 的物理标度错误)")
print(f"  3. 当前 U_cipher 为复振幅场 (complex64), 无法直接加载!")
print(f"     SLM 只能调制相位或振幅之一, 需要转换:")
print(f"       选项A (纯相位SLM): 仅用相位  arg(U_cipher) -> 0~255 灰度")
print(f"       选项B (纯振幅SLM): 仅用振幅  |U_cipher| -> 0~255 灰度")
print(f"       选项C (复振幅, 双SLM): 振幅SLM + 相位SLM 串联")
print(f"       选项D (纯相位全息图): Lee hologram / 叠加离轴载波")
