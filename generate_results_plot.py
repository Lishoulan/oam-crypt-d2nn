# -*- coding: utf-8 -*-
"""
生成精美的结果汇总图 (results.png)
==================================
组合: 原始明文 + 密文 + 解密结果 + PSNR + SLM 加载图
"""
import os
import sys
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from oam_crypt_d2nn import (
    CONFIG, generate_rpp, encrypt_batch, MNISTQuadDataset,
    OAM_Crypt_D2NN, build_target_grid, calculate_psnr
)


def main():
    device = torch.device(CONFIG["device"])
    torch.manual_seed(42)
    np.random.seed(42)

    ckpt = sys.argv[1] if len(sys.argv) > 1 else "oam_crypt_dnn_epoch_20.pth"

    # 1. 加载模型
    transform = transforms.Compose([transforms.ToTensor()])
    full_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    mnist_test = Subset(full_test, range(8))
    test_dataset = MNISTQuadDataset(mnist_test, img_size=CONFIG["size"] // 2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    rpp_system = generate_rpp(CONFIG["size"], device)
    model = OAM_Crypt_D2NN(
        size=CONFIG["size"], num_layers=CONFIG["num_layers"],
        wavelength=CONFIG["wavelength"], pixel_size=CONFIG["pixel_size"],
        z_layer=CONFIG["z_layer"], z0=CONFIG["z0"], rpp=rpp_system,
        oam_keys=CONFIG["l_auth"]
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # 2. 加密 + 解密
    batch_imgs = next(iter(test_loader)).to(device)
    target = build_target_grid(batch_imgs, device, size=CONFIG["size"])
    U_cipher = encrypt_batch(
        batch_imgs, CONFIG["l_auth"], rpp_system,
        CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
        size=CONFIG["size"]
    )

    with torch.no_grad():
        pred = model(U_cipher)
        psnr = calculate_psnr(pred, target).item()

    # 3. 错误密钥测试 (RPP 错误)
    rpp_wrong = generate_rpp(CONFIG["size"], device)
    cipher_wrong = encrypt_batch(
        batch_imgs, CONFIG["l_auth"], rpp_wrong,
        CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
        size=CONFIG["size"]
    )
    with torch.no_grad():
        pred_wrong = model(cipher_wrong)

    # 4. 转换为 numpy
    size = CONFIG["size"]
    half = size // 2
    target_np = target[0].cpu().numpy()
    cipher_amp = torch.abs(U_cipher[0]).cpu().numpy()
    cipher_phase = torch.angle(U_cipher[0]).cpu().numpy()
    pred_np = pred[0].clamp(0, 1).cpu().numpy()
    pred_wrong_np = pred_wrong[0].clamp(0, 1).cpu().numpy()

    # 5. 生成 SLM 加载图 (8-bit)
    gray = np.clip((cipher_phase + np.pi) / (2 * np.pi) * 255 + 0.5, 0, 255).astype(np.uint8)

    # 6. 可视化
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(20, 14))

    # 标题
    fig.suptitle(
        f'OAM-Crypt-D2NN 双密钥全光多用户加解密系统\n'
        f'分辨率 {size}×{size} | 波长 {CONFIG["wavelength"]*1e9:.0f}nm | '
        f'纯相位 SLM 兼容 | PSNR = {psnr:.2f} dB',
        fontsize=18, fontweight='bold', y=0.98
    )

    # 使用 GridSpec 布局
    gs = fig.add_gridspec(3, 5, hspace=0.35, wspace=0.25,
                          top=0.90, bottom=0.04, left=0.05, right=0.95)

    # ======== 行 1: 加密流程 ========
    # 列 1: 原始明文 (4 个数字拼图)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.imshow(target_np, cmap='gray', vmin=0, vmax=1)
    ax1.set_title(f'① 原始明文 (4 张 MNIST)\n左上=l1 | 右上=l2 | 左下=l3 | 右下=l4',
                  fontsize=11, fontweight='bold')
    ax1.axis('off')
    # 象限分割线
    ax1.axhline(y=half, color='r', linewidth=1, linestyle='--', alpha=0.5)
    ax1.axvline(x=half, color='r', linewidth=1, linestyle='--', alpha=0.5)

    # 列 2-3: 密文振幅
    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(cipher_amp, cmap='gray')
    ax2.set_title(f'② 密文 |U| (振幅)\n散斑噪声 (SLM 丢弃)', fontsize=11, fontweight='bold')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.axis('off')

    # 列 4: 密文相位 (= SLM 加载)
    ax3 = fig.add_subplot(gs[0, 3])
    im3 = ax3.imshow(cipher_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax3.set_title(f'③ 密文 arg(U) = SLM 加载\n(纯相位 SLM 唯一加载)', fontsize=11, fontweight='bold')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    ax3.axis('off')

    # 列 5: 8-bit 灰度图
    ax4 = fig.add_subplot(gs[0, 4])
    ax4.imshow(gray, cmap='gray', vmin=0, vmax=255)
    ax4.set_title(f'④ SLM 8-bit 灰度图\n(直接加载到 SLM)', fontsize=11, fontweight='bold')
    ax4.axis('off')

    # ======== 行 2: 解密结果 ========
    # 列 1-2: 解密全图 (正确密钥)
    ax5 = fig.add_subplot(gs[1, 0:2])
    ax5.imshow(pred_np, cmap='gray', vmin=0, vmax=1)
    ax5.set_title(f'⑤ 正确密钥解密 (PSNR = {psnr:.2f} dB)\n'
                  f'OAM + RPP 双密钥匹配', fontsize=11, fontweight='bold', color='green')
    ax5.axis('off')
    ax5.axhline(y=half, color='g', linewidth=1, linestyle='--', alpha=0.5)
    ax5.axvline(x=half, color='g', linewidth=1, linestyle='--', alpha=0.5)

    # 列 3: 解密 4 通道放大
    ax6 = fig.add_subplot(gs[1, 2])
    quads = [
        pred_np[0:half, 0:half], pred_np[0:half, half:size],
        pred_np[half:size, 0:half], pred_np[half:size, half:size]
    ]
    combined = np.concatenate([
        np.concatenate([quads[0], quads[1]], axis=1),
        np.concatenate([quads[2], quads[3]], axis=1)
    ], axis=0)
    ax6.imshow(combined, cmap='gray', vmin=0, vmax=1)
    ax6.set_title('⑥ 解密 4 通道放大', fontsize=11, fontweight='bold')
    ax6.axis('off')

    # 列 4: 错误密钥解密 (应为噪声)
    ax7 = fig.add_subplot(gs[1, 3])
    ax7.imshow(pred_wrong_np, cmap='gray', vmin=0, vmax=1)
    ax7.set_title('⑦ 错误密钥解密\n(RPP 不匹配 → 全黑)',
                  fontsize=11, fontweight='bold', color='red')
    ax7.axis('off')

    # 列 5: 1920×1080 SLM 全屏图
    ax8 = fig.add_subplot(gs[1, 4])
    canvas = np.zeros((1080, 1920), dtype=np.uint8)
    x0_s = (1920 - size) // 2
    canvas[0:size, x0_s:x0_s+size] = gray
    ax8.imshow(canvas, cmap='gray', vmin=0, vmax=255)
    ax8.set_title(f'⑧ SLM 全屏加载图\n1920×1080 (Holoeye PLUTO)', fontsize=11, fontweight='bold')
    rect = Rectangle((x0_s, 0), size, size, linewidth=3,
                     edgecolor='r', facecolor='none')
    ax8.add_patch(rect)
    ax8.text(x0_s + size//2, -50, f'{size}×{size} 全息图',
             color='r', ha='center', fontsize=9, fontweight='bold')
    ax8.axis('off')

    # ======== 行 3: 安全性统计 ========
    # 列 1-2: 解密能量对比 (正确 vs 错误)
    ax9 = fig.add_subplot(gs[2, 0:2])
    energy_correct = np.mean(pred_np)
    energy_wrong = np.mean(pred_wrong_np)
    bars = ax9.bar(['正确密钥\n(OAM+RPP 匹配)', '错误密钥\n(RPP 不匹配)'],
                    [energy_correct, energy_wrong],
                    color=['green', 'red'], alpha=0.7, edgecolor='black')
    ax9.set_ylabel('解密平均能量', fontsize=12)
    ax9.set_title(f'⑨ 密钥敏感性测试\n'
                  f'安全比 = {energy_wrong/energy_correct:.4f} (越低越安全)',
                  fontsize=11, fontweight='bold')
    for bar, val in zip(bars, [energy_correct, energy_wrong]):
        ax9.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                 f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax9.grid(True, alpha=0.3, axis='y')

    # 列 3-5: 系统参数表
    ax10 = fig.add_subplot(gs[2, 2:5])
    ax10.axis('off')
    table_data = [
        ['分辨率', f'{size} × {size}'],
        ['波长 λ', f'{CONFIG["wavelength"]*1e9:.0f} nm'],
        ['像素尺寸', f'{CONFIG["pixel_size"]*1e6:.1f} μm'],
        ['传播距离 z0', f'{CONFIG["z0"]} m'],
        ['OAM 授权密钥', f'l ∈ {CONFIG["l_auth"]}'],
        ['OAM 错误密钥', f'l ∈ {CONFIG["l_wrong"]}'],
        ['训练轮次', f'{20} epoch (warmup)'],
        ['U-Net 通道', f'{12} (4real+4imag+4phase), mid={CONFIG["mid_ch"]}'],
        ['纯相位 SLM PSNR', f'{psnr:.2f} dB'],
        ['Security Ratio', f'{energy_wrong/energy_correct:.4f}'],
    ]
    table = ax10.table(cellText=table_data,
                       colLabels=['参数', '数值'],
                       cellLoc='left',
                       loc='center',
                       colWidths=[0.35, 0.55])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    # 标题样式
    for j in range(2):
        table[0, j].set_facecolor('#4a90e2')
        table[0, j].set_text_props(color='white', fontweight='bold')
    # 交替行颜色
    for i in range(1, len(table_data)+1):
        for j in range(2):
            if i % 2 == 0:
                table[i, j].set_facecolor('#f0f0f0')
    ax10.set_title('⑩ 系统参数汇总', fontsize=12, fontweight='bold', pad=15)

    plt.savefig('results.png', dpi=120, bbox_inches='tight', facecolor='white')
    print(f"已生成: results.png")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SecurityRatio: {energy_wrong/energy_correct:.4f}")


if __name__ == "__main__":
    main()
