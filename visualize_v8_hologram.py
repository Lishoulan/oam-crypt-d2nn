# -*- coding: utf-8 -*-
"""
v8 Lee Hologram 可视化增强版
- 降低偏置 R = 0 dB (R = max|U|), 让载波条纹满量程
- 单独显示 Lee 全息图(大图)
- 对比 v6 振幅物体 vs v8 复振幅 OAM 物体的 Lee 编码差异
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


def lee_hologram_encode(U_cipher, carrier_period=8, ref_offset_db=0.0):
    """Lee hologram: H = arg(R + U·exp(i·2π·f0·x))
    ref_offset_db: R = max|U| × 10^(dB/20)
        0 dB  -> R = max|U|      (满量程调制, 条纹最清晰, 但信号非线性大)
        6 dB  -> R = 2×max|U|    (弱信号近似 R>>|U|, 线性好但条纹弱)
    """
    B, H, W = U_cipher.shape
    device = U_cipher.device
    amp_max = torch.max(torch.abs(U_cipher)).item()
    R = amp_max * (10 ** (ref_offset_db / 20))
    f0 = 1.0 / carrier_period
    x_pix = torch.arange(W, device=device, dtype=torch.float32)
    carrier_phase = 2 * np.pi * f0 * x_pix
    signal_shifted = U_cipher * torch.exp(1j * carrier_phase).unsqueeze(0).unsqueeze(0)
    total_field = R + signal_shifted
    H_phase = torch.angle(total_field)
    return H_phase, R, f0


def main():
    device = torch.device(CONFIG['device'])
    torch.manual_seed(42)
    np.random.seed(42)

    # 加载 v8 stage1 模型
    l_auth = [-25, 25]
    z_list = [0.10, 0.55]
    rpp = generate_rpp(CONFIG['size'], device, generator=torch.Generator(device).manual_seed(42))
    theta_max = np.deg2rad(CONFIG['theta_max_deg'])

    ckpt = torch.load("oam_crypt_v8_stage1_best.pth", map_location=device, weights_only=False)
    model = OAM_Crypt_D2NN(
        size=CONFIG['size'], num_layers=CONFIG['num_layers'],
        wavelength=CONFIG['wavelength'], pixel_size=CONFIG['pixel_size'],
        z_layer=CONFIG['z_layer'], z0=CONFIG['z0'], rpp=rpp,
        oam_keys=l_auth, z_list=z_list,
        obj_encoding=CONFIG['obj_encoding'],
        theta_max=theta_max,
        slm_aware=True,
        use_channel_attn=True, mid_ch=CONFIG['mid_ch'],
        iterative_refine=False, oam_freq_filter=True,
        use_polar_conv=True, polar_n_r=32, polar_n_theta=96,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 取样本并加密
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
    print(f"U_cipher: {tuple(U_cipher.shape)} max|U|={torch.max(torch.abs(U_cipher)).item():.3f}")

    # 三种偏置对比: 0 dB / 6 dB / -6 dB
    cases = [
        ("0 dB (R=max|U|, 满量程)", 0.0),
        ("6 dB (R=2×max|U|, 弱信号)", 6.0),
        ("-6 dB (R=0.5×max|U|, 过调制)", -6.0),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))

    for row, (label, db) in enumerate(cases):
        H_phase, R, f0 = lee_hologram_encode(U_cipher, carrier_period=8, ref_offset_db=db)
        H_np = H_phase[0].cpu().numpy()
        gray = np.clip((H_np + np.pi) / (2 * np.pi) * 255 + 0.5, 0, 255).astype(np.uint8)

        # 取中心 540×540 裁切
        c = 270
        H_crop = H_np[540-c:540+c, 540-c:540+c]
        gray_crop = gray[540-c:540+c, 540-c:540+c]

        # [row, 0] 相位图
        im0 = axes[row, 0].imshow(H_crop, cmap='gray', vmin=-np.pi, vmax=np.pi)
        axes[row, 0].set_title(f"{label}\nH_phase 中心裁切\n范围 [{H_np.min():.2f}, {H_np.max():.2f}] rad",
                               fontsize=12)
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046)
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])

        # [row, 1] 8-bit 灰度
        axes[row, 1].imshow(gray_crop, cmap='gray', vmin=0, vmax=255)
        axes[row, 1].set_title(f"8-bit 灰度 SLM 加载图\n(中心 540×540 裁切)", fontsize=12)
        axes[row, 1].set_xticks([]); axes[row, 1].set_yticks([])

        # [row, 2] 频谱 (+1 级位置标注)
        H_slm = torch.exp(1j * H_phase[0]).cpu().numpy()
        spec = np.abs(np.fft.fftshift(np.fft.fft2(H_slm)))
        spec_log = np.log1p(spec / spec.max() * 1000)  # 归一化让 +1 级可见
        im2 = axes[row, 2].imshow(spec_log, cmap='hot')
        f0_idx = int(round(f0 * 1080))
        axes[row, 2].axvline(x=540 + f0_idx, color='cyan', linestyle='--', alpha=0.7, label=f'+1 级 (x={540+f0_idx})')
        axes[row, 2].axvline(x=540 - f0_idx, color='yellow', linestyle='--', alpha=0.7, label=f'-1 级')
        axes[row, 2].set_title(f"频谱 (归一化 log)\n载波 f0={f0:.3f} cyc/pix", fontsize=12)
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046)
        axes[row, 2].legend(loc='upper right', fontsize=9)
        axes[row, 2].set_xticks([]); axes[row, 2].set_yticks([])

        print(f"[{label}] R={R:.3f} | H_phase 范围 [{H_np.min():.3f}, {H_np.max():.3f}] | "
              f"灰度 [{gray.min()}, {gray.max()}]")

    plt.suptitle('v8 PolarHNN Lee Hologram — 三种偏置对比 (Stage 1, 2 通道, OAM l=±25)',
                 fontsize=15, fontweight='bold', y=0.999)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = "v8_hologram_compare.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n[保存] {out_path}")

    # 额外保存一张"满量程" 1080x1080 全图 Lee hologram
    H_full, R_full, f0_full = lee_hologram_encode(U_cipher, carrier_period=8, ref_offset_db=0.0)
    gray_full = np.clip((H_full[0].cpu().numpy() + np.pi) / (2 * np.pi) * 255 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(gray_full).save("slm_output/slm_hologram_v8_full_modulation.png")

    # 单独的"典型 Lee hologram" 大图(中心 540x540 裁切, 0 dB 偏置)
    H_0db = H_full[0].cpu().numpy()
    c = 270
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    im = ax2.imshow(H_0db[540-c:540+c, 540-c:540+c], cmap='gray', vmin=-np.pi, vmax=np.pi)
    ax2.set_title(f'v8 Lee Hologram (满量程 0 dB 偏置, 中心 540×540)\n'
                  f'清晰可见的载波条纹 (8 pix/cyc) + 调制', fontsize=14, fontweight='bold')
    ax2.set_xticks([]); ax2.set_yticks([])
    plt.colorbar(im, ax=ax2, fraction=0.046, label='相位 (rad)')
    plt.tight_layout()
    plt.savefig("v8_hologram_typical.png", dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[保存] v8_hologram_typical.png (满量程典型 Lee 全息图)")


if __name__ == "__main__":
    main()
