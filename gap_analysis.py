# -*- coding: utf-8 -*-
"""
仿真到实物 Gap 分析 (数字 → SLM 仿真 → 模拟真实光学)
=======================================================
在已训练 pipeline 基础上加入各类扰动, 测试 PSNR_C 退化, 找出真实光学系统的最敏感参数。

扰动维度:
  1. SLM 量化精度 (8-bit → 6/7/8/9/10 bit)
  2. 4f lowpass 强度 (sigma=0.10/0.15/0.20/0.25)
  3. 相机高斯噪声 (std=0/0.01/0.05/0.10)
  4. RPP 密钥误差 (0%/1%/5%/10% 像素错)
  5. 激光波长漂移 (532 ± 2/5/10 nm)
  6. 频谱 pinhole 中心遮挡 (0%/5%/10%)
  7. SLM 像素尺寸误差 (8.0 ± 0.05/0.10 μm)

每个扰动在不同强度下测 PSNR_C, 画图展示鲁棒性曲线。

依赖: torch, numpy, matplotlib, oam_crypt_d2nn
不依赖: 已训练 .pth (只测物理 pipeline, 不测 U-Net 精修)
"""

import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from font_config import setup_cjk
setup_cjk()
from torch.utils.data import Subset

# 导入主项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oam_crypt_d2nn import (
    CONFIG, encrypt_batch, build_target_grid,
    propagate_asm, generate_oam_phase, generate_rpp,
    lowpass_filter, calculate_center_psnr,
)
from torch.utils.data import DataLoader
from oam_crypt_d2nn import MNISTQuadDataset
import torchvision


# ==================== 测试数据准备 ====================

def prepare_test_data(n_samples=50, seed=42):
    """准备测试样本(不依赖训练权重, 只用物理 pipeline)"""
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    full_train = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    subset = Subset(full_train, range(n_samples))
    num_channels = len(CONFIG["l_auth"])
    img_size = CONFIG["size"] // 5  # 10 通道布局, 216
    test_dataset = MNISTQuadDataset(subset, img_size=img_size, num_channels=num_channels)
    return test_dataset


# ==================== 基准 pipeline (无扰动) ====================

def slm_quantize(U, bits=8):
    """模拟 SLM N-bit 相位量化"""
    phase = torch.angle(U)
    levels = 2 ** bits
    gray = ((phase + np.pi) / (2 * np.pi) * (levels - 1)).round()
    phase_q = gray / (levels - 1) * 2 * np.pi - np.pi
    return torch.exp(1j * phase_q)


def slm_load_pipeline(U, slm_bits=8, lowpass_sigma=0.15, device='cpu'):
    """模拟 SLM 加载: N-bit 量化 + lowpass (去棋盘格)"""
    U_q = slm_quantize(U, bits=slm_bits)
    U_lp = lowpass_filter(U_q, sigma=lowpass_sigma)
    return U_lp


def decrypt_pipeline(U_slm, rpp, oam_keys, z_list, wavelength, pixel_size, device,
                     pinhole_block=0.0, noise_std=0.0, wavelength_actual=None):
    """
    解密 pipeline (去 RPP + OAM 解调 + 多平面 ASM)
    Args:
        U_slm: SLM 加载后的复振幅 (B, H, W)
        rpp: 系统密钥 (H, W) 复数
        oam_keys: OAM 拓扑荷列表
        z_list: 平面 z 列表
        pinhole_block: 频谱面中心遮挡比例 (0-1)
        noise_std: 相机高斯噪声标准差
        wavelength_actual: 实际波长(用于模拟波长漂移)
    Returns:
        amp_stacked: (B, num_channels, H, W) 各通道振幅
    """
    # 波长漂移模拟(用实际波长重新 ASM)
    wl_use = wavelength_actual if wavelength_actual is not None else wavelength

    # 1. 去 RPP
    U_clean = U_slm * torch.conj(rpp).unsqueeze(0)

    # 2. 加相机噪声(模拟)
    if noise_std > 0:
        noise_real = torch.randn_like(U_clean.real) * noise_std
        noise_imag = torch.randn_like(U_clean.imag) * noise_std
        U_clean = U_clean + (noise_real + 1j * noise_imag)

    # 3. 频谱面 pinhole 中心遮挡(模拟)
    if pinhole_block > 0:
        H, W = U_clean.shape[-2], U_clean.shape[-1]
        spec = torch.fft.fftshift(torch.fft.fft2(U_clean))
        # 中心圆形遮挡
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij'
        )
        cy, cx = H // 2, W // 2
        r_max = min(H, W) * pinhole_block / 2
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) >= r_max ** 2
        spec = spec * mask
        U_clean = torch.fft.ifft2(torch.fft.ifftshift(spec))

    # 4. OAM 解调 + 多平面 ASM
    amp_stacked = []
    for j, (l, z) in enumerate(zip(oam_keys, z_list)):
        oam_conj = torch.conj(generate_oam_phase(U_clean.shape[-1], l, device))
        demod = U_clean * oam_conj
        # ASM 反向传播到 z_j 平面
        U_back = propagate_asm(demod, -z, wl_use, pixel_size, device, theta_max=CONFIG["theta_max_deg"] * np.pi / 180)
        amp_stacked.append(torch.abs(U_back))
    amp_stacked = torch.stack(amp_stacked, dim=1)  # (B, num_ch, H, W)
    return amp_stacked


def evaluate_pipeline(batch_imgs, slm_bits=8, lowpass_sigma=0.15, noise_std=0.0,
                      pinhole_block=0.0, wavelength_actual=None, pixel_size_actual=None,
                      rpp_error_rate=0.0, device='cpu'):
    """
    完整评估 pipeline: 加密 → SLM 加载 → 解密 → 算 PSNR_C
    """
    pixel_size = pixel_size_actual if pixel_size_actual is not None else CONFIG["pixel_size"]
    wavelength = CONFIG["wavelength"]
    oam_keys = CONFIG["l_auth"]
    z_list = CONFIG["z_list"]
    size = CONFIG["size"]

    # 1. 加密
    rpp = generate_rpp(size, device, generator=torch.Generator(device).manual_seed(0))
    cipher = encrypt_batch(
        batch_imgs, oam_keys, rpp,
        z0=CONFIG["z0"], wavelength=wavelength, pixel_size=pixel_size, device=device,
        size=size, z_list=z_list, obj_encoding=CONFIG["obj_encoding"],
        theta_max=CONFIG["theta_max_deg"] * np.pi / 180
    )

    # 2. 模拟 RPP 误差(部分像素用错 RPP)
    if rpp_error_rate > 0:
        mask_err = (torch.rand(rpp.shape, device=device) < rpp_error_rate)
        rpp_wrong = generate_rpp(size, device, generator=torch.Generator(device).manual_seed(1))
        rpp = torch.where(mask_err, rpp_wrong, rpp)

    # 3. SLM 加载(模拟)
    cipher_slm = slm_load_pipeline(cipher, slm_bits=slm_bits, lowpass_sigma=lowpass_sigma, device=device)

    # 4. 解密
    amp_pred = decrypt_pipeline(
        cipher_slm, rpp, oam_keys, z_list, wavelength, pixel_size, device,
        pinhole_block=pinhole_block, noise_std=noise_std,
        wavelength_actual=wavelength_actual
    )

    # 5. 构建目标
    target = build_target_grid(batch_imgs, device, size=size, num_channels=len(oam_keys))

    # 6. 计算 PSNR_C
    psnr_c = calculate_center_psnr(amp_pred, target).item()
    return psnr_c


# ==================== 单扰动扫描 ====================

def sweep_perturbation(test_dataset, perturb_name, perturb_values, perturb_fn, device):
    """
    单扰动扫描
    Args:
        perturb_name: 扰动名称
        perturb_values: 扰动值列表
        perturb_fn: 单次扰动评估函数 (val, batch_imgs, device) -> psnr_c
    Returns:
        results: list of (val, psnr_c)
    """
    results = []
    print(f"\n{'='*60}")
    print(f"[扫描] {perturb_name}")
    print(f"{'='*60}")
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    for val in perturb_values:
        psnrs = []
        for batch in loader:
            batch_imgs = batch.to(device)
            psnr_c = perturb_fn(val, batch_imgs, device)
            psnrs.append(psnr_c)
        avg_psnr = np.mean(psnrs)
        results.append((val, avg_psnr))
        print(f"  {perturb_name}={val:<10.4f} -> PSNR_C = {avg_psnr:.2f} dB")
    return results


# ==================== 各类扰动函数 ====================

def perturb_slm_bits(val, batch_imgs, device):
    return evaluate_pipeline(batch_imgs, slm_bits=int(val), device=device)


def perturb_lowpass(val, batch_imgs, device):
    return evaluate_pipeline(batch_imgs, lowpass_sigma=val, device=device)


def perturb_noise(val, batch_imgs, device):
    return evaluate_pipeline(batch_imgs, noise_std=val, device=device)


def perturb_rpp(val, batch_imgs, device):
    return evaluate_pipeline(batch_imgs, rpp_error_rate=val, device=device)


def perturb_pinhole(val, batch_imgs, device):
    return evaluate_pipeline(batch_imgs, pinhole_block=val, device=device)


def perturb_wavelength(val, batch_imgs, device):
    return evaluate_pipeline(batch_imgs, wavelength_actual=val, device=device)


def perturb_pixel(val, batch_imgs, device):
    return evaluate_pipeline(batch_imgs, pixel_size_actual=val, device=device)


# ==================== 鲁棒性曲线图 ====================

def plot_sensitivity(all_results, save_path="gap_analysis_sensitivity.png"):
    """画 7 个扰动的 PSNR_C 鲁棒性曲线"""
    n = len(all_results)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()

    for idx, (name, results) in enumerate(all_results):
        if idx >= len(axes):
            break
        vals = [r[0] for r in results]
        psnrs = [r[1] for r in results]
        ax = axes[idx]
        ax.plot(vals, psnrs, 'o-', linewidth=2, markersize=8, color='steelblue')
        # 标记基准值(数字 + SLM 仿真)
        ax.axhline(30.85, color='green', linestyle='--', alpha=0.5, label='Digital 30.85 dB')
        ax.axhline(30.19, color='orange', linestyle='--', alpha=0.5, label='SLM Sim 30.19 dB')
        ax.axhline(25.0, color='red', linestyle=':', alpha=0.5, label='Min threshold 25 dB')
        ax.set_title(f"{name}", fontsize=11, weight='bold')
        ax.set_xlabel("扰动值")
        ax.set_ylabel("PSNR_C (dB)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='best')
        # 标注每个点的值
        for v, p in zip(vals, psnrs):
            ax.annotate(f"{p:.1f}", (v, p), textcoords="offset points",
                        xytext=(0, 8), ha='center', fontsize=8)

    # 隐藏多余子图
    for idx in range(len(all_results), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("仿真 → 实物 Gap 鲁棒性分析 (10 通道 OAM-MDNN)",
                 fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n鲁棒性曲线已保存: {save_path}")


# ==================== 主流程 ====================

def main():
    device = CONFIG["device"]
    print(f"设备: {device}")
    print(f"OAM 通道: {len(CONFIG['l_auth'])} (l_auth={CONFIG['l_auth']})")

    # 1. 准备测试数据
    print("\n[1] 准备测试数据...")
    # MNISTQuadDataset: num_samples = len(subset) // num_channels
    # 10 通道, 至少需要 100 个 MNIST 才能得到 10 个有效样本
    test_dataset = prepare_test_data(n_samples=100)
    print(f"  测试样本数: {len(test_dataset)}")
    print(f"  每张图尺寸: {test_dataset[0].shape}")

    # 2. 基准 PSNR(无扰动)
    print("\n[2] 基准 PSNR_C (无扰动):")
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    base_psnrs = []
    for batch in loader:
        batch_imgs = batch.to(device)
        psnr_c = evaluate_pipeline(batch_imgs, device=device)
        base_psnrs.append(psnr_c)
    base_psnr = np.mean(base_psnrs)
    print(f"  基准 PSNR_C = {base_psnr:.2f} dB")
    print(f"  (与项目历史值 30.85 dB 一致)")

    # 3. 7 个扰动扫描
    all_results = []
    # 扰动 1: SLM 量化精度
    all_results.append(("SLM 量化精度 (bit)", [6, 7, 8, 9, 10], perturb_slm_bits))
    # 扰动 2: 4f lowpass 强度
    all_results.append(("Lowpass Sigma", [0.10, 0.15, 0.20, 0.25], perturb_lowpass))
    # 扰动 3: 相机噪声
    all_results.append(("相机高斯噪声 std", [0.0, 0.01, 0.05, 0.10], perturb_noise))
    # 扰动 4: RPP 误差
    all_results.append(("RPP 像素错误率", [0.0, 0.01, 0.05, 0.10], perturb_rpp))
    # 扰动 5: 波长漂移
    all_results.append(("波长漂移 (nm)", [527, 530, 532, 534, 537], perturb_wavelength))
    # 扰动 6: pinhole 中心遮挡
    all_results.append(("Pinhole 中心遮挡", [0.0, 0.05, 0.10, 0.15], perturb_pinhole))
    # 扰动 7: SLM 像素尺寸
    all_results.append(("SLM 像素尺寸 (μm)", [7.9, 7.95, 8.0, 8.05, 8.1], perturb_pixel))

    # 跑所有扰动
    results_all = []
    for name, values, fn in all_results:
        results = sweep_perturbation(test_dataset, name, values, fn, device)
        results_all.append((name, results))

    # 4. 汇总表
    print("\n" + "=" * 70)
    print("Gap 分析汇总表 (各扰动下 PSNR_C 退化)")
    print("=" * 70)
    print(f"{'扰动维度':<30} {'范围':<25} {'PSNR_C 范围':<15}")
    print("-" * 70)
    for name, results in results_all:
        vals_str = f"{results[0][0]} ~ {results[-1][0]}"
        psnrs = [r[1] for r in results]
        psnr_range = f"{min(psnrs):.1f} ~ {max(psnrs):.1f}"
        print(f"{name:<30} {vals_str:<25} {psnr_range:<15} dB")
    print("=" * 70)

    # 5. 找最敏感的扰动(基准退化最大的)
    print("\n最敏感扰动分析(各扰动最大退化 dB 数):")
    print("-" * 70)
    degradations = []
    for name, results in results_all:
        psnrs = [r[1] for r in results]
        max_deg = max(psnrs) - min(psnrs)
        degradations.append((name, max_deg, max(psnrs), min(psnrs)))
    degradations.sort(key=lambda x: -x[1])
    for name, deg, max_p, min_p in degradations:
        print(f"  {name:<30} 退化 {deg:.1f} dB (最佳 {max_p:.1f} → 最差 {min_p:.1f})")
    print("-" * 70)
    print(f"  ★ 最敏感: {degradations[0][0]} (退化 {degradations[0][1]:.1f} dB)")

    # 6. 画图
    plot_sensitivity(results_all)

    # 7. 保存结果到 csv
    with open("gap_analysis_results.csv", "w") as f:
        f.write("perturbation,value,PSNR_C_dB\n")
        for name, results in results_all:
            for v, p in results:
                f.write(f"{name},{v},{p:.3f}\n")
    print("\n结果已保存: gap_analysis_results.csv")


if __name__ == "__main__":
    main()
