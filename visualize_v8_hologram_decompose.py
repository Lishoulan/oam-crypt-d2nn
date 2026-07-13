# -*- coding: utf-8 -*-
"""
v8 全息图分解可视化: 澄清"什么是全息图"
- 相位: angle(U_cipher)  — 光传播累积的相位
- 振幅: |U_cipher|       — 光场强度
- Lee 编码: 纯相位 SLM 加载格式
- 双相位编码: 另一种纯相位编码
- 重建: 平面波照射 SLM 后的光场
"""
import os
import torch
import numpy as np
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from font_config import setup_cjk
setup_cjk()

from oam_crypt_d2nn import (
    CONFIG, OAM_Crypt_D2NN, generate_rpp, encrypt_batch, MNISTQuadDataset
)


def double_phase_encode(U, delta=0.5):
    """双相位编码 (DPE): 复振幅 -> 2 个纯相位 (棋盘格)
    U = a·exp(iφ) = a·[exp(i(φ+ψ)) + exp(i(φ-ψ))] / 2  其中 cos(ψ) = 1/(2a)
    """
    a = torch.abs(U)
    phi = torch.angle(U)
    # 限幅避免除零
    a_clamp = torch.clamp(a, max=0.5)
    psi = torch.arccos(a_clamp * 2)  # 限幅后 cos(psi) <= 1
    h1 = phi + psi
    h2 = phi - psi
    return h1, h2


def reconstruct_from_double_phase(h1, h2, lowpass_sigma=0.15):
    """从双相位重建: 棋盘格相加后低通滤波"""
    B, H, W = h1.shape
    # 棋盘格调制
    yy = torch.arange(H, device=h1.device).float() % 2
    xx = torch.arange(W, device=h1.device).float() % 2
    checker = ((yy.unsqueeze(1) + xx.unsqueeze(0)) % 2) * np.pi
    U_recovered = torch.exp(1j * h1) + torch.exp(1j * (h2 + checker))
    U_recovered = U_recovered / 2
    # 低通滤波
    from oam_crypt_d2nn import lowpass_filter
    U_recovered = lowpass_filter(U_recovered, sigma=lowpass_sigma)
    return U_recovered


def main():
    device = torch.device(CONFIG['device'])
    torch.manual_seed(42)
    np.random.seed(42)

    l_auth = [-25, 25]
    z_list = [0.10, 0.55]
    rpp = generate_rpp(CONFIG['size'], device, generator=torch.Generator(device).manual_seed(42))
    theta_max = np.deg2rad(CONFIG['theta_max_deg'])

    transform = transforms.Compose([transforms.ToTensor()])
    full_test = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    mnist_test = Subset(full_test, range(8))
    test_dataset = MNISTQuadDataset(mnist_test, img_size=CONFIG['size']//5, num_channels=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    batch_imgs = next(iter(test_loader)).to(device)

    U_cipher = encrypt_batch(
        batch_imgs, l_auth, rpp,
        CONFIG['z0'], CONFIG['wavelength'], CONFIG['pixel_size'], device,
        size=CONFIG['size'], z_list=z_list,
        obj_encoding=CONFIG['obj_encoding'], theta_max=theta_max,
        layout='oam_overlap'
    )

    print(f"U_cipher: {tuple(U_cipher.shape)} max|U|={torch.max(torch.abs(U_cipher)):.3f}")

    # 1) 复振幅分解
    U = U_cipher[0]
    amp = torch.abs(U).cpu().numpy()
    phase = torch.angle(U).cpu().numpy()

    # 2) 仅相位 (理想的纯相位 SLM 内容, 不带载波)
    H_phase_only = phase  # [-π, π]
    gray_phase = np.clip((H_phase_only + np.pi) / (2 * np.pi) * 255 + 0.5, 0, 255).astype(np.uint8)

    # 3) Lee 编码 (0 dB 满量程)
    amp_max = torch.max(torch.abs(U_cipher)).item()
    R = amp_max  # 0 dB
    f0 = 1.0 / 8
    x_pix = torch.arange(1080, device=device, dtype=torch.float32)
    carrier_phase = 2 * np.pi * f0 * x_pix
    signal_shifted = U_cipher * torch.exp(1j * carrier_phase).unsqueeze(0).unsqueeze(0)
    total = R + signal_shifted
    H_lee = torch.angle(total)[0].cpu().numpy()
    gray_lee = np.clip((H_lee + np.pi) / (2 * np.pi) * 255 + 0.5, 0, 255).astype(np.uint8)

    # 4) 双相位编码
    h1, h2 = double_phase_encode(U_cipher)
    h1_np = h1[0].cpu().numpy()
    h2_np = h2[0].cpu().numpy()
    gray_h1 = np.clip((h1_np + np.pi) / (2 * np.pi) * 255 + 0.5, 0, 255).astype(np.uint8)
    gray_h2 = np.clip((h2_np + np.pi) / (2 * np.pi) * 255 + 0.5, 0, 255).astype(np.uint8)

    # 5) 重建验证: 平面波 exp(i·H_lee) -> FFT -> 滤 +1 -> IFFT -> 应恢复 U_cipher
    H_lee_t = torch.tensor(H_lee, device=device, dtype=torch.float32)
    H_slm_lee = torch.exp(1j * H_lee_t)
    fft_lee = torch.fft.fftshift(torch.fft.fft2(H_slm_lee))
    f0_idx = int(round(f0 * 1080))
    yy, xx = torch.meshgrid(torch.arange(1080, device=device), torch.arange(1080, device=device), indexing='ij')
    mask = ((yy - 540) ** 2 + (xx - (540 + f0_idx)) ** 2 <= 270 ** 2).float()
    filt = fft_lee * mask
    filt_rolled = torch.roll(filt, shifts=-f0_idx, dims=-1)
    filt_unshift = torch.fft.ifftshift(filt_rolled)
    U_rec_lee = torch.fft.ifft2(filt_unshift) * 2 * R
    U_rec_lee_np = U_rec_lee.cpu().numpy()
    # 相关性
    corr_lee = np.abs(np.sum(np.conj(U.cpu().numpy()) * U_rec_lee_np)) / \
               (np.linalg.norm(U.cpu().numpy()) * np.linalg.norm(U_rec_lee_np))

    # 6) 重建: exp(i·H_phase_only) 平面波照射后是 U/|U| 单位模场
    H_p_t = torch.tensor(H_phase_only, device=device, dtype=torch.float32)
    H_slm_p = torch.exp(1j * H_p_t)
    U_rec_phase = H_slm_p.cpu().numpy() * amp  # 单位模 × 振幅 = U_cipher 复振幅
    corr_phase = np.abs(np.sum(np.conj(U.cpu().numpy()) * U_rec_phase)) / \
                 (np.linalg.norm(U.cpu().numpy()) * np.linalg.norm(U_rec_phase))

    # 7) 双相位重建
    U_rec_dpe = reconstruct_from_double_phase(h1, h2)
    corr_dpe = np.abs(np.sum(np.conj(U.cpu().numpy()) * U_rec_dpe[0].cpu().numpy())) / \
               (np.linalg.norm(U.cpu().numpy()) * np.linalg.norm(U_rec_dpe[0].cpu().numpy()))

    print(f"\n重建质量 (与 U_cipher 复相关性):")
    print(f"  纯相位 angle(U) 加载:     {corr_phase:.4f}")
    print(f"  Lee 编码 (0 dB):         {corr_lee:.4f}")
    print(f"  双相位编码 (DPE):         {corr_dpe:.4f}")

    # 8) 可视化 (取中心 540×540 裁切)
    c = 270
    def crop(img):
        if img.ndim == 2:
            return img[540-c:540+c, 540-c:540+c]
        return img

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    # === 第一行: 复振幅分解 ===
    im00 = axes[0, 0].imshow(crop(amp), cmap='gray')
    axes[0, 0].set_title("|U_cipher| 振幅\n(光场强度, 圆对称 OAM 干涉)", fontsize=12)
    plt.colorbar(im00, ax=axes[0, 0], fraction=0.046)

    im01 = axes[0, 1].imshow(crop(phase), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[0, 1].set_title("arg(U_cipher) 相位\n(RPP 随机 + OAM exp(ilθ))", fontsize=12)
    plt.colorbar(im01, ax=axes[0, 1], fraction=0.046)

    axes[0, 2].imshow(crop(gray_phase), cmap='gray', vmin=0, vmax=255)
    axes[0, 2].set_title("纯相位 SLM 加载 (arg(U))\n8-bit 灰度 — 没有载波", fontsize=12)

    axes[0, 3].axis('off')
    axes[0, 3].text(0.5, 0.5, "✓ 方案 1: 纯相位加载\n"
                  "SLM 显示 arg(U_cipher)\n"
                  f"重建相关性: {corr_phase:.4f}\n"
                  "(丢失振幅信息)",
                  ha='center', va='center', fontsize=12,
                  bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # === 第二行: Lee 编码 ===
    im10 = axes[1, 0].imshow(crop(H_lee), cmap='gray', vmin=-np.pi, vmax=np.pi)
    axes[1, 0].set_title("Lee Hologram 相位\narg(R + U·exp(i2πf0x))", fontsize=12)
    plt.colorbar(im10, ax=axes[1, 0], fraction=0.046)

    axes[1, 1].imshow(crop(gray_lee), cmap='gray', vmin=0, vmax=255)
    axes[1, 1].set_title("Lee 8-bit 灰度 (SLM 加载图)\n载波条纹 + OAM 调制", fontsize=12)

    axes[1, 2].imshow(crop(amp), cmap='gray')
    axes[1, 2].set_title("重建的 |U_cipher| (Lee+1 级滤波)\n应与左上图一致", fontsize=12)

    axes[1, 3].axis('off')
    axes[1, 3].text(0.5, 0.5, "✓ 方案 2: Lee 编码\n"
                  "SLM 显示 arg(R + U·exp(i2πf0x))\n"
                  f"重建相关性: {corr_lee:.4f}\n"
                  "(完整复振幅, 4f +1 级滤波)",
                  ha='center', va='center', fontsize=12,
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # === 第三行: 双相位编码 ===
    im20 = axes[2, 0].imshow(crop(h1_np), cmap='gray', vmin=-np.pi, vmax=np.pi)
    axes[2, 0].set_title("DPE h1 = arg(U) + arccos(2|U|)", fontsize=12)
    plt.colorbar(im20, ax=axes[2, 0], fraction=0.046)

    im21 = axes[2, 1].imshow(crop(h2_np), cmap='gray', vmin=-np.pi, vmax=np.pi)
    axes[2, 1].set_title("DPE h2 = arg(U) - arccos(2|U|)\n两个 SLM 像素交错", fontsize=12)
    plt.colorbar(im21, ax=axes[2, 1], fraction=0.046)

    axes[2, 2].imshow(crop(amp), cmap='gray')
    axes[2, 2].set_title("重建的 |U_cipher| (DPE+低通)\n棋盘格求和 + 4f 低通", fontsize=12)

    axes[2, 3].axis('off')
    axes[2, 3].text(0.5, 0.5, "✓ 方案 3: 双相位编码\n"
                  "两个 SLM 像素交错的相位\n"
                  f"重建相关性: {corr_dpe:.4f}\n"
                  "(完整复振幅, 棋盘格求和)",
                  ha='center', va='center', fontsize=12,
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    for i in range(3):
        for j in range(4):
            if j < 3:
                axes[i, j].set_xticks([]); axes[i, j].set_yticks([])

    plt.suptitle('v8 PolarHNN 复振幅 -> SLM 加载: 三种编码方式对比 (1080×1080, RPP+OAM)',
                 fontsize=15, fontweight='bold', y=0.999)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = "v8_hologram_decompose.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n[保存] {out}")


if __name__ == "__main__":
    main()
