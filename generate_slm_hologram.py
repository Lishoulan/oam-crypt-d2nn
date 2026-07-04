# -*- coding: utf-8 -*-
"""
SLM 纯相位全息图生成器 (Lee Hologram / 离轴参考光法)
=====================================================
原理: 纯相位 SLM 只能加载 arg(H), 但通过叠加离轴参考光
      H(x,y) = arg( U_cipher(x,y) + R·exp(i·2π·f0·x) )
      可在 +1 级衍射中完整恢复复振幅 U_cipher。

适用: Holoeye PLUTO 系列 (1920×1080, 8.0 μm, 纯相位)
输出: 8-bit 灰度 PNG/BMP, 可直接加载到 SLM 控制软件
"""
import os
import torch
import numpy as np
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from oam_crypt_d2nn import (
    CONFIG, generate_rpp, encrypt_batch, MNISTQuadDataset
)


# ==========================================
# SLM 配置 (Holoeye PLUTO-2.1)
# ==========================================
SLM_CONFIG = {
    "width": 1920,           # SLM 水平像素数
    "height": 1080,           # SLM 垂直像素数
    "pixel_size": 8.0e-6,     # 8 μm (与训练代码一致)
    "carrier_period_pix": 8,  # 载波周期 (像素/周期), 8pix/cyc 充分分离 0/±1 级
    "ref_offset_db": 6.0,     # 参考光偏置: R = max|U| × 10^(dB/20), 6dB=2×, 保线性近似
    "output_dir": "slm_output",
}


def lee_hologram_encode(U_cipher, carrier_period=8, ref_offset_db=6.0):
    """
    Lee hologram 编码: 将复振幅 U 编码为纯相位 H ∈ [-π, π]

    H(x,y) = arg( R + U(x,y)·exp(i·2π·f0·x) )

    物理含义 (注意方向: 参考光在零级, 信号被搬到 +1 级):
      - R                       : 实常数偏置 (DC 分量, 在零级)
      - U·exp(i·2π·f0·x)        : 信号被载波搬到 +f0 频率
      - arg(·)                  : 取相位 (SLM 只能加载相位)
      - +1 级滤波 (在 +f0 附近) 可恢复完整 U

    数学 (R >> |U| 时, 弱信号近似):
      exp(i·arg(R + U·e^{i2πf0x})) ≈ 1 + U·e^{i2πf0x}/R
      => FFT 后零级有 1 (DC), +1 级有 U/R

    参数:
      U_cipher           : (B, H, W) complex64 复振幅场
      carrier_period     : 载波周期 (像素/周期), 越小频谱分离越好但采样越紧
      ref_offset_db      : 参考光振幅偏置 (dB), 6dB = 2×max|U|, 防过调制且保线性
    返回:
      H_phase            : (B, H, W) float32, 范围 [-π, π]
      R                  : 参考光振幅标量
      f0                 : 载波频率 (cycles/pixel)
    """
    B, H, W = U_cipher.shape
    device = U_cipher.device

    # 参考光振幅: 大于信号最大振幅, 避免过调制且保证线性近似
    amp_max = torch.max(torch.abs(U_cipher)).item()
    R = amp_max * (10 ** (ref_offset_db / 20))

    # 载波频率: f0 = 1 / 周期
    f0 = 1.0 / carrier_period

    # x 方向载波相位 (1D 沿 x 线性增长)
    x_pix = torch.arange(W, device=device, dtype=torch.float32)
    carrier_phase = 2 * np.pi * f0 * x_pix  # (W,)

    # 关键: U 乘载波 (搬到 +f0), R 为实常数 (在零级)
    signal_shifted = U_cipher * torch.exp(1j * carrier_phase).unsqueeze(0).unsqueeze(0)
    total_field = R + signal_shifted  # R 在 0 频, U 在 +f0 频
    H_phase = torch.angle(total_field)  # (B,H,W) ∈ [-π, π]

    return H_phase, R, f0


def phase_to_uint8(H_phase):
    """
    将相位 [-π, π] 映射为 8-bit 灰度 [0, 255]
    公式: gray = round( (phase + π) / (2π) × 255 )
    """
    H_np = H_phase.cpu().numpy()
    gray = np.clip((H_np + np.pi) / (2 * np.pi) * 255 + 0.5, 0, 255).astype(np.uint8)
    return gray


def zero_pad_to_slm(gray_2d, slm_h, slm_w):
    """
    将 128×128 全息图居中 zero-pad 到 SLM 原生分辨率
    周围填 0 (=相位 π, 不影响光路, 因为 0 灰度=相位 0 也是均匀的, 这里用 0=相位 0)
    """
    H_small, W_small = gray_2d.shape
    canvas = np.zeros((slm_h, slm_w), dtype=np.uint8)  # 0 = 相位 0 (均匀, 不产生衍射)
    y0 = (slm_h - H_small) // 2
    x0 = (slm_w - W_small) // 2
    canvas[y0:y0 + H_small, x0:x0 + W_small] = gray_2d
    return canvas, (y0, x0)


def reconstruct_from_lee_hologram(H_phase, R, f0):
    """
    数值验证: 从 Lee hologram 重建复振幅, 用于检验编码质量
    步骤:
      1. SLM 实际加载场 H_slm = exp(i·H_phase)
      2. FFT, 用圆形窗口提取 +1 级 (中心在 f0 处)
      3. 把 +1 级 roll 回中心 (去载波)
      4. IFFT 得到复振幅, 乘 R 恢复 U
    弱信号近似下: H_slm ≈ 1 + U·e^{i2πf0x}/R
      => FFT(H_slm) 在零级有 N·M (DC), 在 +f0 处有 U/R 的频谱
    """
    H_slm = torch.exp(1j * H_phase)
    H_fft = torch.fft.fft2(H_slm)
    H_fft_shift = torch.fft.fftshift(H_fft, dim=(-2, -1))

    B, h, w = H_phase.shape
    f0_idx = int(round(f0 * w))  # 载波在频域的索引偏移
    cx, cy = w // 2, h // 2

    # +1 级圆形滤波窗口 (足够大以覆盖 U 的宽频谱, 散斑是宽频信号)
    filter_radius = h // 4  # 半径 32 像素, 覆盖大部分信号能量
    yy, xx = torch.meshgrid(
        torch.arange(h, device=H_phase.device),
        torch.arange(w, device=H_phase.device),
        indexing='ij'
    )
    mask_plus1 = ((yy - cy) ** 2 + (xx - (cx + f0_idx)) ** 2 <= filter_radius ** 2).float()
    mask_plus1 = mask_plus1.unsqueeze(0)

    # 提取 +1 级, 然后把窗口 roll 回中心 (相当于去载波)
    filtered = H_fft_shift * mask_plus1
    filtered_rolled = torch.roll(filtered, shifts=-f0_idx, dims=-1)
    filtered_unshift = torch.fft.ifftshift(filtered_rolled, dim=(-2, -1))
    U_recovered = torch.fft.ifft2(filtered_unshift)

    # 弱信号近似推导: H_slm = exp(i·arg(R + U·e^{i2πf0x}))
    #                        ≈ 1 + i·Im(U·e^{i2πf0x})/R
    #                        = 1 + (U·e^{i2πf0x} - U*·e^{-i2πf0x}) / (2R)
    # => +1 级 (在 +f0 处) 信号是 U/(2R), 乘 2R 还原
    U_recovered = U_recovered * 2 * R
    return U_recovered


def main():
    os.makedirs(SLM_CONFIG["output_dir"], exist_ok=True)
    device = torch.device(CONFIG["device"])
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. 加载测试数据 + RPP
    transform = transforms.Compose([transforms.ToTensor()])
    full_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    mnist_test = Subset(full_test, range(8))
    test_dataset = MNISTQuadDataset(mnist_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    rpp_system = generate_rpp(CONFIG["size"], device)

    # 2. 取一个样本并加密
    batch_imgs = next(iter(test_loader)).to(device)
    U_cipher = encrypt_batch(
        batch_imgs, CONFIG["l_auth"], rpp_system,
        CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
        z_list=CONFIG["z_list"]
    )  # (1, 128, 128) complex64

    print("=" * 65)
    print("Lee Hologram 编码 (纯相位 SLM 加载格式)")
    print("=" * 65)
    print(f"SLM: {SLM_CONFIG['width']}×{SLM_CONFIG['height']} @ "
          f"{SLM_CONFIG['pixel_size']*1e6:.1f} μm (Holoeye PLUTO)")
    print(f"源 cipher: {tuple(U_cipher.shape)} complex64")
    print(f"载波周期: {SLM_CONFIG['carrier_period_pix']} pix/cyc "
          f"(f0 = {1/SLM_CONFIG['carrier_period_pix']:.4f} cyc/pix = "
          f"{1/(SLM_CONFIG['carrier_period_pix']*SLM_CONFIG['pixel_size']*1e3):.1f} cyc/mm)")
    print(f"参考光偏置: {SLM_CONFIG['ref_offset_db']} dB")

    # 3. Lee hologram 编码
    H_phase, R, f0 = lee_hologram_encode(
        U_cipher,
        carrier_period=SLM_CONFIG["carrier_period_pix"],
        ref_offset_db=SLM_CONFIG["ref_offset_db"]
    )

    print()
    print("编码结果统计:")
    print(f"  H_phase 范围: [{H_phase.min().item():.4f}, {H_phase.max().item():.4f}] rad")
    print(f"  参考光振幅 R = {R:.4f} (信号 max|U| = {torch.max(torch.abs(U_cipher)).item():.4f})")

    # 4. 数值重建验证
    U_recovered = reconstruct_from_lee_hologram(H_phase, R, f0)
    err = torch.abs(U_recovered - U_cipher)
    rel_err = torch.mean(err) / torch.mean(torch.abs(U_cipher))
    corr = torch.abs(torch.sum(U_cipher.conj() * U_recovered)) / \
           (torch.norm(U_cipher) * torch.norm(U_recovered) + 1e-12)

    print()
    print("重建质量验证 (FFT 滤波提取 +1 级):")
    print(f"  相对误差:        {rel_err.item():.4f} ({rel_err.item()*100:.2f}%)")
    print(f"  复振幅相关系数:  {corr.item():.6f} (1.0 = 完美重建)")

    # 5. 转 8-bit 灰度并 zero-pad 到 SLM 分辨率
    gray_small = phase_to_uint8(H_phase[0])  # (128, 128)
    gray_slm, (y0, x0) = zero_pad_to_slm(
        gray_small, SLM_CONFIG["height"], SLM_CONFIG["width"]
    )

    print()
    print("输出文件:")
    print(f"  128×128 原始相位图 (用于仿真验证):")
    print(f"    -> {SLM_CONFIG['output_dir']}/lee_hologram_128.png")
    print(f"  1920×1080 SLM 加载图 (居中 zero-pad):")
    print(f"    -> {SLM_CONFIG['output_dir']}/lee_hologram_SLM_1920x1080.png")
    print(f"  全息图在 SLM 上的位置: x=[{x0}, {x0+128}], y=[{y0}, {y0+128}]")

    # 6. 保存文件
    Image.fromarray(gray_small).save(
        os.path.join(SLM_CONFIG["output_dir"], "lee_hologram_128.png")
    )
    Image.fromarray(gray_slm).save(
        os.path.join(SLM_CONFIG["output_dir"], "lee_hologram_SLM_1920x1080.png")
    )

    # 7. 同时保存振幅图和复振幅实虚部, 便于诊断
    amp_np = torch.abs(U_cipher[0]).cpu().numpy()
    amp_norm = (amp_np / amp_np.max() * 255).astype(np.uint8)
    Image.fromarray(amp_norm).save(
        os.path.join(SLM_CONFIG["output_dir"], "cipher_amplitude_128.png")
    )

    # 8. 保存可视化的对比图
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 复振幅振幅
        im0 = axes[0, 0].imshow(amp_np, cmap='gray')
        axes[0, 0].set_title(f"Cipher |U| (复振幅振幅)\n"
                             f"范围 [{amp_np.min():.4f}, {amp_np.max():.4f}]")
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

        # 复振幅相位
        phase_np = torch.angle(U_cipher[0]).cpu().numpy()
        im1 = axes[0, 1].imshow(phase_np, cmap='twilight', vmin=-np.pi, vmax=np.pi)
        axes[0, 1].set_title(f"Cipher arg(U) (复振幅相位)\n"
                             f"范围 [-π, π]")
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

        # Lee hologram 相位 (SLM 加载内容)
        im2 = axes[1, 0].imshow(gray_small, cmap='gray', vmin=0, vmax=255)
        axes[1, 0].set_title(f"Lee Hologram (SLM 加载)\n"
                             f"8-bit 灰度, 128×128, 载波可见为条纹")
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

        # SLM 完整图
        axes[1, 1].imshow(gray_slm, cmap='gray', vmin=0, vmax=255)
        axes[1, 1].set_title(f"SLM 全屏加载图\n"
                             f"1920×1080, 全息图居中")
        # 标出全息图位置
        rect = plt.Rectangle((x0, y0), 128, 128, linewidth=2,
                             edgecolor='r', facecolor='none')
        axes[1, 1].add_patch(rect)
        axes[1, 1].text(x0 + 64, y0 - 30, "Hologram", color='r',
                        ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(SLM_CONFIG["output_dir"], "lee_hologram_overview.png"),
                    dpi=120, bbox_inches='tight')
        print(f"  对比图概览:")
        print(f"    -> {SLM_CONFIG['output_dir']}/lee_hologram_overview.png")
    except Exception as e:
        print(f"  (matplotlib 可视化跳过: {e})")

    print()
    print("=" * 65)
    print("SLM 加载说明")
    print("=" * 65)
    print(f"1. 将 lee_hologram_SLM_1920x1080.png 加载到 Holoeye PLUTO 控制软件")
    print(f"2. SLM 工作波长设为 633 nm (He-Ne 激光)")
    print(f"3. 在 SLM 后放置傅里叶透镜 (焦距 f), 在其后焦面观察衍射谱")
    print(f"4. 用光阑选取 +1 级 (位于 f·λ·f0 = {0.1*633e-9*1/SLM_CONFIG['carrier_period_pix']/SLM_CONFIG['pixel_size']*1e3:.2f} mm 偏移处, "
          f"假设 f=100mm)")
    print(f"5. +1 级即恢复的 U_cipher, 后续接入解密 D2NN 光路")


if __name__ == "__main__":
    main()
