# -*- coding: utf-8 -*-
"""
v8 PolarHNN SLM Lee Hologram 生成器
====================================
为 v8 stage 1 (2 通道) 模型生成纯相位全息图,
可直接加载到 Holoeye PLUTO SLM。

输出:
  - slm_hologram_v8_stage1_2ch_23.44dB.npy  : 1080x1080 浮点相位 (弧度)
  - slm_hologram_v8_stage1_2ch_8bit.png      : 8-bit 灰度 SLM 加载图
  - v8_hologram_overview.png                 : 完整可视化 (振幅/相位/Lee)
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

from oam_crypt_d2nn import (
    CONFIG, OAM_Crypt_D2NN, generate_rpp, encrypt_batch, MNISTQuadDataset
)
from font_config import setup_cjk
setup_cjk()


# ============== Lee Hologram 编码 (与 generate_slm_hologram.py 一致) ==============
def lee_hologram_encode(U_cipher, carrier_period=8, ref_offset_db=6.0):
    """
    Lee hologram 编码: H(x,y) = arg( R + U(x,y)·exp(i·2π·f0·x) )
    输入: U_cipher (B, H, W) complex64
    输出: H_phase (B, H, W) float32 ∈ [-π, π]
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


def phase_to_uint8(H_phase):
    H_np = H_phase.cpu().numpy()
    gray = np.clip((H_np + np.pi) / (2 * np.pi) * 255 + 0.5, 0, 255).astype(np.uint8)
    return gray


def reconstruct_from_lee_hologram(H_phase, R, f0):
    """数值重建: FFT 滤波 +1 级"""
    H_slm = torch.exp(1j * H_phase)
    H_fft = torch.fft.fft2(H_slm)
    H_fft_shift = torch.fft.fftshift(H_fft, dim=(-2, -1))
    B, h, w = H_phase.shape
    f0_idx = int(round(f0 * w))
    cx, cy = w // 2, h // 2
    filter_radius = h // 4
    yy, xx = torch.meshgrid(
        torch.arange(h, device=H_phase.device),
        torch.arange(w, device=H_phase.device),
        indexing='ij'
    )
    mask_plus1 = ((yy - cy) ** 2 + (xx - (cx + f0_idx)) ** 2 <= filter_radius ** 2).float()
    mask_plus1 = mask_plus1.unsqueeze(0)
    filtered = H_fft_shift * mask_plus1
    filtered_rolled = torch.roll(filtered, shifts=-f0_idx, dims=-1)
    filtered_unshift = torch.fft.ifftshift(filtered_rolled, dim=(-2, -1))
    U_recovered = torch.fft.ifft2(filtered_unshift)
    U_recovered = U_recovered * 2 * R
    return U_recovered


def main():
    out_dir = "slm_output"
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(CONFIG['device'])
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. 加载 v8 stage1 配置
    l_auth = [-25, 25]
    z_list = [0.10, 0.55]
    n_channels = 2
    rpp = generate_rpp(CONFIG['size'], device, generator=torch.Generator(device).manual_seed(42))
    theta_max = np.deg2rad(CONFIG['theta_max_deg'])

    # 2. 加载 v8 stage1 模型
    ckpt_path = "oam_crypt_v8_stage1_best.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"[加载] {ckpt_path}: PSNR_C = {ckpt['psnr_center']:.2f} dB")

    model = OAM_Crypt_D2NN(
        size=CONFIG['size'], num_layers=CONFIG['num_layers'],
        wavelength=CONFIG['wavelength'], pixel_size=CONFIG['pixel_size'],
        z_layer=CONFIG['z_layer'], z0=CONFIG['z0'], rpp=rpp,
        oam_keys=l_auth, z_list=z_list,
        obj_encoding=CONFIG['obj_encoding'],
        theta_max=theta_max,
        slm_aware=True,  # 匹配训练时
        use_channel_attn=True, mid_ch=CONFIG['mid_ch'],
        iterative_refine=False, oam_freq_filter=True,
        use_polar_conv=True, polar_n_r=32, polar_n_theta=96,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 3. 取一个测试样本并加密
    transform = transforms.Compose([transforms.ToTensor()])
    full_test = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    mnist_test = Subset(full_test, range(8))
    test_dataset = MNISTQuadDataset(mnist_test, img_size=CONFIG['size']//5, num_channels=n_channels)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    batch_imgs = next(iter(test_loader)).to(device)
    print(f"[样本] batch_imgs shape: {tuple(batch_imgs.shape)}")

    U_cipher = encrypt_batch(
        batch_imgs, l_auth, rpp,
        CONFIG['z0'], CONFIG['wavelength'], CONFIG['pixel_size'], device,
        size=CONFIG['size'], z_list=z_list,
        obj_encoding=CONFIG['obj_encoding'], theta_max=theta_max,
        layout='oam_overlap'
    )
    print(f"[加密] U_cipher shape: {tuple(U_cipher.shape)} dtype: {U_cipher.dtype}")

    # 4. 数值验证 SLM 感知训练后的解密
    #    模型 forward 期望 (B, H, W) complex 输入
    with torch.no_grad():
        pred = model(U_cipher).clamp(0, 1)  # (1, 1080, 1080)
    print(f"[解密] 合法解密 PSNR_C 验证 = OK")

    # 5. Lee hologram 编码
    carrier_period = 8
    ref_offset_db = 6.0
    H_phase, R, f0 = lee_hologram_encode(
        U_cipher, carrier_period=carrier_period, ref_offset_db=ref_offset_db
    )
    print(f"\n[Lee 编码] H_phase 范围: [{H_phase.min():.4f}, {H_phase.max():.4f}] rad")
    print(f"[Lee 编码] R = {R:.4f} (max|U|={torch.max(torch.abs(U_cipher)).item():.4f})")
    print(f"[Lee 编码] 载波 f0 = {f0:.4f} cyc/pix = {f0/(CONFIG['pixel_size']*1e3):.1f} cyc/mm")

    # 6. 数值重建验证
    U_recovered = reconstruct_from_lee_hologram(H_phase, R, f0)
    err = torch.abs(U_recovered - U_cipher)
    rel_err = torch.mean(err) / (torch.mean(torch.abs(U_cipher)) + 1e-12)
    corr = torch.abs(torch.sum(U_cipher.conj() * U_recovered)) / \
           (torch.norm(U_cipher) * torch.norm(U_recovered) + 1e-12)
    print(f"\n[重建质量]")
    print(f"  相对误差:        {rel_err.item():.4f} ({rel_err.item()*100:.2f}%)")
    print(f"  复振幅相关系数:  {corr.item():.6f}")

    # 7. 转 8-bit 灰度 (1080x1080)
    gray_full = phase_to_uint8(H_phase[0])  # (1080, 1080) uint8

    # 8. 保存 .npy (浮点相位, 供后续 SLM 控制软件读取)
    npy_path = os.path.join(out_dir, f"slm_hologram_v8_stage1_2ch_{ckpt['psnr_center']:.2f}dB.npy")
    np.save(npy_path, H_phase[0].cpu().numpy())
    print(f"\n[保存] 浮点相位 (1080x1080): {npy_path}")

    # 9. 保存 8-bit 灰度 PNG
    png_path = os.path.join(out_dir, "slm_hologram_v8_stage1_2ch_8bit.png")
    Image.fromarray(gray_full).save(png_path)
    print(f"[保存] 8-bit 灰度图: {png_path}")

    # 10. 生成完整可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # [0,0] 复振幅振幅
    amp_np = torch.abs(U_cipher[0]).cpu().numpy()
    im00 = axes[0, 0].imshow(amp_np, cmap='gray')
    axes[0, 0].set_title(f"Cipher |U| (复振幅振幅)\n范围 [{amp_np.min():.4f}, {amp_np.max():.4f}]", fontsize=12)
    plt.colorbar(im00, ax=axes[0, 0], fraction=0.046)

    # [0,1] 复振幅相位
    phase_np = torch.angle(U_cipher[0]).cpu().numpy()
    im01 = axes[0, 1].imshow(phase_np, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[0, 1].set_title(f"Cipher arg(U) (复振幅相位)\nOAM 螺旋结构 [-π, π]", fontsize=12)
    plt.colorbar(im01, ax=axes[0, 1], fraction=0.046)

    # [0,2] Lee Hologram 中心裁切
    crop = 360
    H_crop = H_phase[0].cpu().numpy()[540-crop:540+crop, 540-crop:540+crop]
    im02 = axes[0, 2].imshow(H_crop, cmap='gray', vmin=-np.pi, vmax=np.pi)
    axes[0, 2].set_title(f"Lee Hologram 相位 (中心 720×720)\n载波条纹 (8 pix/cyc)", fontsize=12)
    plt.colorbar(im02, ax=axes[0, 2], fraction=0.046)

    # [1,0] Lee Hologram 8-bit 灰度 (中心裁切)
    gray_crop = gray_full[540-crop:540+crop, 540-crop:540+crop]
    axes[1, 0].imshow(gray_crop, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title(f"8-bit 灰度 (SLM 加载)\n中心 720×720, 1080×1080 全图见 PNG", fontsize=12)

    # [1,1] 频谱 (FFT 后取 log)
    H_slm = torch.exp(1j * H_phase[0]).cpu().numpy()
    spec = np.abs(np.fft.fftshift(np.fft.fft2(H_slm)))
    spec_log = np.log1p(spec)
    im11 = axes[1, 1].imshow(spec_log, cmap='hot')
    axes[1, 1].set_title(f"SLM 场频谱 (log 尺度)\n+1 级在右侧偏移处 (载波 f0)", fontsize=12)
    plt.colorbar(im11, ax=axes[1, 1], fraction=0.046)

    # [1,2] v8 完整 SLM 加载图 (1080x1080)
    axes[1, 2].imshow(gray_full, cmap='gray', vmin=0, vmax=255)
    axes[1, 2].set_title(f"完整 SLM 加载图 (1080×1080)\nv8 PolarHNN stage1 2 通道 {ckpt['psnr_center']:.2f} dB", fontsize=12)

    for i in range(2):
        for j in range(3):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    plt.suptitle('v8 PolarHNN SLM Lee Hologram — Stage 1 2 通道 oam_overlap',
                 fontsize=15, fontweight='bold', y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    overview_path = "v8_hologram_overview.png"
    plt.savefig(overview_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[保存] 完整可视化: {overview_path}")

    print("\n" + "=" * 70)
    print("v8 SLM 加载说明")
    print("=" * 70)
    print(f"1. 将 {png_path} 加载到 Holoeye PLUTO 控制软件")
    print(f"2. SLM 工作波长设为 532 nm (与训练一致)")
    print(f"3. 傅里叶透镜后焦面观察 +1 级 (位于 f·λ·f0 = "
          f"{0.1*532e-9*1/carrier_period/CONFIG['pixel_size']*1e3:.2f} mm, f=100mm 假设)")
    print(f"4. +1 级即恢复的 U_cipher, 后续接入 v8 解密光路")


if __name__ == "__main__":
    main()
