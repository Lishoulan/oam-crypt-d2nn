# -*- coding: utf-8 -*-
"""
v8 端到端物理链路验证:
Lee Hologram SLM 加载 -> 4f 傅里叶变换 -> +1 级滤波 -> D2NN 解密 -> 重建 MNIST

关键问题: Lee 全息图 + 4f + D2NN 能否恢复原图?
"""
import os
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from font_config import setup_cjk
setup_cjk()

from oam_crypt_d2nn import (
    CONFIG, OAM_Crypt_D2NN, generate_rpp, encrypt_batch,
    build_target_grid, calculate_center_psnr, MNISTQuadDataset,
    propagate_asm
)


def lee_hologram(U_cipher, carrier_period=8, ref_offset_db=0.0):
    B, H, W = U_cipher.shape
    device = U_cipher.device
    amp_max = torch.max(torch.abs(U_cipher)).item()
    R = amp_max * (10 ** (ref_offset_db / 20))
    f0 = 1.0 / carrier_period
    x_pix = torch.arange(W, device=device, dtype=torch.float32)
    carrier = 2 * np.pi * f0 * x_pix
    shifted = U_cipher * torch.exp(1j * carrier).unsqueeze(0).unsqueeze(0)
    H = torch.angle(R + shifted)
    return H, R, f0


def four_f_reconstruct(H_phase, R, f0, filter_radius, prop_z=0.0,
                       wavelength=532e-9, pixel_size=8.0e-6, device='cuda'):
    """
    4f 系统重建:
      1. SLM 加载: H_slm = exp(i·H_phase)  (纯相位)
      2. 傅里叶透镜: FFT  → 频域面
      3. +1 级圆形滤波
      4. 逆向去载波 (roll)
      5. IFFT → 物面
      (可选) 6. 二次相位补偿: 物面 * exp(i·πr²/λf) 但 FFT/IFFT 已天然做
    """
    H, W = H_phase.shape[-2:]
    # SLM 加载场
    H_slm = torch.exp(1j * H_phase)
    # 4f: 傅里叶变换 (FFT 在焦面)
    F_H = torch.fft.fftshift(torch.fft.fft2(H_slm), dim=(-2, -1))
    # +1 级位置
    f0_idx = int(round(f0 * W))
    cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(
        torch.arange(H, device=H_phase.device),
        torch.arange(W, device=H_phase.device),
        indexing='ij'
    )
    # 圆形窗口
    mask = ((yy - cy) ** 2 + (xx - (cx + f0_idx)) ** 2 <= filter_radius ** 2).float()
    # 提取 +1 级
    F_plus1 = F_H * mask.unsqueeze(0)
    # 滚回中心
    F_unshift = torch.roll(F_plus1, shifts=-f0_idx, dims=-1)
    F_unshift = torch.fft.ifftshift(F_unshift, dim=(-2, -1))
    # 逆变换
    U_recovered = torch.fft.ifft2(F_unshift)
    # 弱信号近似: H_slm ≈ 1 + U·e^{i2πf0x}/R
    # 提取的 +1 级 (经 roll 回到中心) 是 U/R, 乘 2R 还原 (因为 1/(2R) 系数)
    U_recovered = U_recovered * 2 * R
    return U_recovered, mask


def main():
    device = torch.device(CONFIG['device'])
    torch.manual_seed(42)
    np.random.seed(42)

    l_auth = [-25, 25]
    z_list = [0.10, 0.55]
    rpp = generate_rpp(CONFIG['size'], device, generator=torch.Generator(device).manual_seed(42))
    theta_max = np.deg2rad(CONFIG['theta_max_deg'])

    # 加载 v8 stage1 模型
    ckpt = torch.load("oam_crypt_v8_stage1_best.pth", map_location=device, weights_only=False)
    model = OAM_Crypt_D2NN(
        size=CONFIG['size'], num_layers=CONFIG['num_layers'],
        wavelength=CONFIG['wavelength'], pixel_size=CONFIG['pixel_size'],
        z_layer=CONFIG['z_layer'], z0=CONFIG['z0'], rpp=rpp,
        oam_keys=l_auth, z_list=z_list,
        obj_encoding=CONFIG['obj_encoding'],
        theta_max=theta_max,
        slm_aware=True,  # 关键: SLM 感知, 已包含 8-bit 量化训练
        use_channel_attn=True, mid_ch=CONFIG['mid_ch'],
        iterative_refine=False, oam_freq_filter=True,
        use_polar_conv=True, polar_n_r=32, polar_n_theta=96,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 准备多个测试样本
    transform = transforms.Compose([transforms.ToTensor()])
    full_test = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    mnist_test = Subset(full_test, range(0, 40))
    test_dataset = MNISTQuadDataset(mnist_test, img_size=CONFIG['size']//5, num_channels=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 收集 4 个样本
    samples = []
    targets = []
    for i, b in enumerate(test_loader):
        if i >= 4: break
        b = b.to(device)
        samples.append(b)
        targets.append(build_target_grid(b, device, size=CONFIG['size'], layout='oam_overlap'))
    print(f"准备了 {len(samples)} 个测试样本")

    # 加密所有样本
    ciphers = []
    for b in samples:
        c = encrypt_batch(
            b, l_auth, rpp,
            CONFIG['z0'], CONFIG['wavelength'], CONFIG['pixel_size'], device,
            size=CONFIG['size'], z_list=z_list,
            obj_encoding=CONFIG['obj_encoding'], theta_max=theta_max,
            layout='oam_overlap'
        )
        ciphers.append(c)

    # ===== 三种 baseline 解密 =====
    with torch.no_grad():
        # 1. 数字直接解密 (无 SLM 模拟)
        psnr_digital = []
        for c, t in zip(ciphers, targets):
            p = model(c).clamp(0, 1)
            psnr_digital.append(calculate_center_psnr(p, t).item())
        print(f"\n[Baseline 1] 数字直接解密 (无 SLM): {np.mean(psnr_digital):.2f} dB (平均)")

        # 2. SLM 8-bit 量化直接 (不经过 Lee, 直接 angle+8bit)
        psnr_slm8bit = []
        for c, t in zip(ciphers, targets):
            phase = torch.angle(c)
            gray = ((phase + np.pi) / (2 * np.pi) * 255).round()
            phase_q = gray / 255.0 * 2 * np.pi - np.pi
            c_slm = torch.exp(1j * phase_q)  # 纯相位 SLM
            p = model(c_slm).clamp(0, 1)
            psnr_slm8bit.append(calculate_center_psnr(p, t).item())
        print(f"[Baseline 2] SLM 8-bit 量化直接: {np.mean(psnr_slm8bit):.2f} dB (平均)")

        # 3. SLM 感知训练 (模型内部已经模拟 8bit) — 这就是 ckpt 的 23.44 dB
        psnr_slm_aware = psnr_slm8bit  # 一样
        print(f"[Baseline 3] SLM-aware 训练 (ckpt): {ckpt['psnr_center']:.2f} dB")

    # ===== Lee + 4f + D2NN 链路 =====
    print(f"\n{'='*70}")
    print("Lee Hologram + 4f + D2NN 端到端测试")
    print(f"{'='*70}")

    results = []
    for bias_db in [0.0, 6.0, -3.0]:
        for filter_r in [135, 270, 540]:
            psnrs = []
            for c, t in zip(ciphers, targets):
                # 1. Lee 编码
                H, R, f0 = lee_hologram(c, carrier_period=8, ref_offset_db=bias_db)
                # 2. SLM 加载 + 8-bit 量化
                phase_slm = torch.angle(torch.exp(1j * H))  # 实际就是 H
                gray = ((phase_slm + np.pi) / (2 * np.pi) * 255).round()
                phase_q = gray / 255.0 * 2 * np.pi - np.pi
                H_quantized = phase_q
                # 3. SLM 场 = exp(i·H_quantized)
                # 4. 4f 傅里叶变换 + +1 级滤波 + 重建
                U_rec, mask = four_f_reconstruct(H_quantized, R, f0, filter_radius=filter_r)
                # 5. D2NN 解密
                with torch.no_grad():
                    p = model(U_rec).clamp(0, 1)
                psnrs.append(calculate_center_psnr(p, t).item())
            mean_psnr = np.mean(psnrs)
            results.append((bias_db, filter_r, mean_psnr, psnrs))
            print(f"  偏置 {bias_db:+.0f} dB, 滤波半径 {filter_r:3d} pix: PSNR = {mean_psnr:.2f} dB  (样本: {[f'{p:.1f}' for p in psnrs]})")

    # ===== 找最佳配置 =====
    best = max(results, key=lambda x: x[2])
    print(f"\n[最佳] 偏置 {best[0]:+.0f} dB, 滤波半径 {best[1]} pix: PSNR = {best[2]:.2f} dB")

    # ===== 物理解释 =====
    print(f"\n{'='*70}")
    print("物理解释")
    print(f"{'='*70}")
    print(f"v8 cipher 特征: max|U|={torch.max(torch.abs(ciphers[0])).item():.3f}")
    print(f"加密链路: 明文 → OAM×2 模式 → RPP 随机相位 → 自由空间传播")
    print(f"RPP 效应: 频谱能量均匀分布到所有频率, +1 级滤波窗口越小越丢失")
    print(f"")
    print(f"Lee 偏置选择:")
    print(f"  0 dB (R=max|U|): 满量程调制, +1 级能量大, 但非线性失真")
    print(f"  6 dB (R=2×max|U|): 弱信号近似线性, +1 级能量衰减到 1/4")
    print(f"  -3 dB (R=0.7×max|U|): 过调制, 信号折叠, 失真严重")
    print(f"")
    print(f"滤波窗口选择 (相对 f0_idx={int(round(0.125*1080))}):")
    print(f"  135: 紧贴 +1 级中心, 噪声大")
    print(f"  270: 标准 4f 孔径, 平衡")
    print(f"  540: 大窗口, 包含几乎全部信号能量")

    # ===== 可视化对比 =====
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    # 行 0: 原始链路 (无 Lee)
    for i in range(4):
        # 取中心 216x216 (明文是 (2, 1080, 1080), 中心 108x108 是 1 通道)
        c, t = ciphers[i], targets[i]
        # targets shape: (1, 2, 1080, 1080) — 拼接 2 通道横向
        plain_c0 = t[0, 0, 540-108:540+108, 540-108:540+108].cpu().numpy()
        plain_c1 = t[0, 1, 540-108:540+108, 540-108:540+108].cpu().numpy()
        plain_concat = np.concatenate([plain_c0, plain_c1], axis=1)
        axes[0, i].imshow(plain_concat, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f"样本 {i+1} 明文 (2 通道)" if i == 0 else f"样本 {i+1}", fontsize=10)

    # 行 1: SLM-aware 数字解密 (无 Lee)
    for i in range(4):
        with torch.no_grad():
            p = model(ciphers[i]).clamp(0, 1)
        p_c0 = p[0, 0, 540-108:540+108, 540-108:540+108].cpu().numpy()
        p_c1 = p[0, 1, 540-108:540+108, 540-108:540+108].cpu().numpy()
        p_concat = np.concatenate([p_c0, p_c1], axis=1)
        axes[1, i].imshow(p_concat, cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f"数字解密 (SLM-aware)\n{psnr_digital[i]:.2f} dB" if i == 0 else f"{psnr_digital[i]:.1f} dB", fontsize=10)

    # 行 2: Lee 编码 (0 dB, 最佳滤波) + 4f + D2NN
    bias_db, filter_r, _, _ = best[0], best[1], best[2], best[3]
    for i in range(4):
        c, t = ciphers[i], targets[i]
        H, R, f0 = lee_hologram(c, carrier_period=8, ref_offset_db=bias_db)
        # SLM 8bit 量化
        gray = ((H + np.pi) / (2 * np.pi) * 255).round()
        H_q = gray / 255.0 * 2 * np.pi - np.pi
        # 4f 重建
        U_rec, _ = four_f_reconstruct(H_q, R, f0, filter_radius=filter_r)
        # D2NN
        with torch.no_grad():
            p = model(U_rec).clamp(0, 1)
        p_c0 = p[0, 0, 540-108:540+108, 540-108:540+108].cpu().numpy()
        p_c1 = p[0, 1, 540-108:540+108, 540-108:540+108].cpu().numpy()
        p_concat = np.concatenate([p_c0, p_c1], axis=1)
        axes[2, i].imshow(p_concat, cmap='gray', vmin=0, vmax=1)
        axes[2, i].set_title(f"Lee {bias_db:+.0f} dB + 4f r={filter_r}\n{best[3][i]:.2f} dB" if i == 0 else f"{best[3][i]:.1f} dB", fontsize=10)

    # 行 3: 最佳配置 vs 直接数字
    for i in range(4):
        axes[3, i].axis('off')
    axes[3, 0].text(0.5, 0.5, f"端到端物理链路验证\n\n"
                  f"明文 → 加密 → Lee Hologram\n"
                  f"→ SLM 8-bit 加载\n"
                  f"→ 4f 傅里叶变换 + +1 级滤波\n"
                  f"→ 物面重建 → D2NN 解密\n"
                  f"→ MNIST 数字\n\n"
                  f"最佳配置: 偏置 {best[0]:+.0f} dB, 滤波 {best[1]} pix\n"
                  f"PSNR: {best[2]:.2f} dB\n"
                  f"(SLM-aware 训练 baseline: {np.mean(psnr_slm_aware):.2f} dB)",
                  ha='center', va='center', fontsize=11,
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    for i in range(3):
        for j in range(4):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])

    plt.suptitle('v8 端到端: Lee Hologram → SLM → 4f → D2NN → 重建 MNIST',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = "v8_end_to_end.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n[保存] {out}")


if __name__ == "__main__":
    main()
