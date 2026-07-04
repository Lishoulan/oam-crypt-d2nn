# -*- coding: utf-8 -*-
"""
SLM 加载方案对比测试
====================
测试 3 种纯相位 SLM 加载方案对解密质量的影响:
  方案 A: 完整复振幅 (理想参考, 上界)
  方案 B: 仅相位 arg(U_cipher), 丢振幅 (最简单 SLM 加载)
  方案 C: Lee hologram (离轴载波, 复振幅重建)
"""
import os
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from PIL import Image

from oam_crypt_d2nn import (
    CONFIG, generate_rpp, encrypt_batch, MNISTQuadDataset,
    OAM_Crypt_D2NN, build_target_grid, calculate_psnr, save_security_plot
)


def load_model(device, rpp_system, ckpt_path):
    """加载训练好的解密网络"""
    model = OAM_Crypt_D2NN(
        size=CONFIG["size"], num_layers=CONFIG["num_layers"],
        wavelength=CONFIG["wavelength"], pixel_size=CONFIG["pixel_size"],
        z_layer=CONFIG["z_layer"], z0=CONFIG["z0"], rpp=rpp_system,
        oam_keys=CONFIG["l_auth"]
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def test_scheme(model, U_cipher, scheme, device):
    """
    对 cipher 应用不同的 SLM 加载方案, 送入解密网络测 PSNR
    scheme:
      'full'      : 完整复振幅 (理想参考)
      'phase_only': 仅相位 exp(i·arg(U))  (纯相位 SLM 直接加载)
      'lee'       : Lee hologram + FFT 重建
    """
    if scheme == 'full':
        U_slm = U_cipher  # 完整复振幅
    elif scheme == 'phase_only':
        # 纯相位 SLM 加载: exp(i·arg(U)) = U / |U|, 振幅归一化为 1
        phase = torch.angle(U_cipher)
        U_slm = torch.exp(1j * phase)
    elif scheme == 'lee':
        # Lee hologram: arg(R + U·exp(i·2π·f0·x))
        f0 = 1.0 / 8  # 载波周期 8 像素
        R = torch.max(torch.abs(U_cipher)).item() * 2.0  # 6dB 偏置
        x_pix = torch.arange(U_cipher.shape[-1], device=device, dtype=torch.float32)
        carrier = torch.exp(1j * 2 * np.pi * f0 * x_pix)
        total = R + U_cipher * carrier.unsqueeze(0).unsqueeze(0)
        H_phase = torch.angle(total)
        U_slm = torch.exp(1j * H_phase)  # SLM 加载后的场
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    with torch.no_grad():
        pred = model(U_slm)
    return pred, U_slm


def main():
    device = torch.device(CONFIG["device"])
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. 数据 + RPP + 模型
    transform = transforms.Compose([transforms.ToTensor()])
    full_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    mnist_test = Subset(full_test, range(4000))
    test_dataset = MNISTQuadDataset(mnist_test)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    rpp_system = generate_rpp(CONFIG["size"], device)
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "oam_crypt_dnn_epoch_80.pth"
    model = load_model(device, rpp_system, ckpt)
    print(f"使用 checkpoint: {ckpt}")

    print("=" * 70)
    print("SLM 加载方案对比 (纯相位 SLM 兼容性测试)")
    print("=" * 70)
    print(f"模型: oam_crypt_dnn_epoch_80.pth (PSNR 37.45 dB 训练结果)")
    print()

    schemes = ['full', 'phase_only', 'lee']
    scheme_names = {
        'full': '方案 A: 完整复振幅 (理想上界)',
        'phase_only': '方案 B: 仅相位 arg(U) (纯相位 SLM 直接加载)',
        'lee': '方案 C: Lee hologram (离轴载波重建)',
    }

    psnr_results = {s: [] for s in schemes}
    sample_data = None  # 保存第一个样本用于可视化

    for batch_idx, batch_imgs in enumerate(test_loader):
        batch_imgs = batch_imgs.to(device)
        target = build_target_grid(batch_imgs, device)

        cipher = encrypt_batch(
            batch_imgs, CONFIG["l_auth"], rpp_system,
            CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device
        )

        for scheme in schemes:
            pred, U_slm = test_scheme(model, cipher, scheme, device)
            psnr = calculate_psnr(pred, target).item()
            psnr_results[scheme].append(psnr)

            if batch_idx == 0 and sample_data is None:
                sample_data = {
                    'cipher': cipher[0].cpu(),
                    'target': target[0].cpu(),
                    'pred_full': None,
                    'pred_phase_only': None,
                    'pred_lee': None,
                    'U_slm_phase_only': None,
                    'U_slm_lee': None,
                }

        if batch_idx == 0:
            sample_data['pred_full'], _ = test_scheme(model, cipher, 'full', device)
            sample_data['pred_phase_only'], sample_data['U_slm_phase_only'] = \
                test_scheme(model, cipher, 'phase_only', device)
            sample_data['pred_lee'], sample_data['U_slm_lee'] = \
                test_scheme(model, cipher, 'lee', device)
            sample_data['pred_full'] = sample_data['pred_full'][0].cpu()
            sample_data['pred_phase_only'] = sample_data['pred_phase_only'][0].cpu()
            sample_data['pred_lee'] = sample_data['pred_lee'][0].cpu()
            sample_data['U_slm_phase_only'] = sample_data['U_slm_phase_only'][0].cpu()
            sample_data['U_slm_lee'] = sample_data['U_slm_lee'][0].cpu()

    # 统计结果
    print(f"{'方案':<45} {'平均 PSNR':<12} {'最高':<10} {'最低':<10}")
    print("-" * 77)
    for scheme in schemes:
        psnrs = np.array(psnr_results[scheme])
        print(f"{scheme_names[scheme]:<45} {psnrs.mean():>8.2f} dB  "
              f"{psnrs.max():>6.2f} dB  {psnrs.min():>6.2f} dB")

    print()
    print("=" * 70)
    print("关键诊断")
    print("=" * 70)
    full_psnr = np.mean(psnr_results['full'])
    po_psnr = np.mean(psnr_results['phase_only'])
    lee_psnr = np.mean(psnr_results['lee'])
    print(f"完整复振幅 (理想):     {full_psnr:.2f} dB")
    print(f"仅相位 SLM (方案 B):   {po_psnr:.2f} dB  (损失 {full_psnr - po_psnr:.2f} dB)")
    print(f"Lee hologram (方案 C): {lee_psnr:.2f} dB  (损失 {full_psnr - lee_psnr:.2f} dB)")
    print()
    print("结论:")
    if po_psnr > 25:
        print(f"  ✓ 方案 B (仅相位) 可行! PSNR {po_psnr:.2f} dB > 25 dB")
        print(f"    原因: 解密网络 forward 中第一步 amp_mean 归一化已对振幅不敏感")
        print(f"    推荐: 直接加载 arg(U_cipher) 到 SLM, 无需 Lee hologram 复杂编码")
    else:
        print(f"  ✗ 方案 B (仅相位) 不可行! PSNR {po_psnr:.2f} dB 太低")
        print(f"    需要考虑 Lee hologram 或重新设计加密流程")
    if lee_psnr > full_psnr - 3:
        print(f"  ✓ 方案 C (Lee hologram) 接近理想: 损失仅 {full_psnr - lee_psnr:.2f} dB")
    else:
        print(f"  ✗ 方案 C (Lee hologram) 损失过大: {full_psnr - lee_psnr:.2f} dB")

    # 保存可视化对比
    print()
    print("生成可视化对比图...")
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(4, 5, figsize=(20, 16))

    # 行 1: 加载到 SLM 的场 (强度)
    titles_row1 = [
        "Cipher |U| (完整复振幅)",
        "SLM 方案B: |exp(i·arg(U))| = 1 (恒定)",
        "SLM 方案C: |Lee hologram| (含载波条纹)",
        "Cipher arg(U) (相位, 三方案共用)",
        "目标明文 (Ground Truth)",
    ]
    imgs_row1 = [
        torch.abs(sample_data['cipher']).numpy(),
        torch.abs(sample_data['U_slm_phase_only']).numpy(),
        torch.abs(sample_data['U_slm_lee']).numpy(),
        torch.angle(sample_data['cipher']).numpy(),
        sample_data['target'].numpy(),
    ]
    cmaps = ['gray', 'gray', 'gray', 'twilight', 'gray']
    vmaxs = [None, 1.0, None, np.pi, 1.0]
    vmins = [0, 0, 0, -np.pi, 0]
    for i, (img, title, cmap, vmax, vmin) in enumerate(zip(imgs_row1, titles_row1, cmaps, vmaxs, vmins)):
        im = axes[0, i].imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0, i].set_title(title, fontsize=10)
        plt.colorbar(im, ax=axes[0, i], fraction=0.046)

    # 行 2-4: 三种方案的解密结果 (4 个象限)
    scheme_preds = [
        ('方案 A: 完整复振幅', sample_data['pred_full']),
        ('方案 B: 仅相位 SLM', sample_data['pred_phase_only']),
        ('方案 C: Lee hologram', sample_data['pred_lee']),
    ]
    for row, (name, pred) in enumerate(scheme_preds, start=1):
        pred_np = pred.clamp(0, 1).numpy()
        # 全图
        axes[row, 0].imshow(pred_np, cmap='gray', vmin=0, vmax=1)
        axes[row, 0].set_title(f"{name}\n解密全图 (128×128)", fontsize=10)
        # 4 个象限
        quads = [
            pred_np[0:64, 0:64], pred_np[0:64, 64:128],
            pred_np[64:128, 0:64], pred_np[64:128, 64:128]
        ]
        for i, q in enumerate(quads):
            axes[row, i+1].imshow(q, cmap='gray', vmin=0, vmax=1)
            axes[row, i+1].set_title(f"通道 {i}", fontsize=10)

    plt.tight_layout()
    out_path = "slm_scheme_comparison.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
