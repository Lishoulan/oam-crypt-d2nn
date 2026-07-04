# -*- coding: utf-8 -*-
"""
SLM 纯相位全息图生成器 (匹配训练好的解密网络)
================================================
由于解密网络 forward 开头已做相位化 exp(i·arg(U)),
SLM 只需直接加载 arg(U_cipher) 即可, 无需 Lee hologram 复杂编码。

输出: 8-bit 灰度 PNG, 可直接加载到 Holoeye PLUTO SLM
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from oam_crypt_d2nn import (
    CONFIG, generate_rpp, encrypt_batch, MNISTQuadDataset,
    OAM_Crypt_D2NN, build_target_grid, calculate_psnr
)


SLM_CONFIG = {
    "width": 1920,
    "height": 1080,
    "pixel_size": 8.0e-6,
    "output_dir": "slm_output_1080",
}


def main():
    device = torch.device(CONFIG["device"])
    torch.manual_seed(42)
    np.random.seed(42)

    ckpt = sys.argv[1] if len(sys.argv) > 1 else "oam_crypt_dnn_epoch_20.pth"
    os.makedirs(SLM_CONFIG["output_dir"], exist_ok=True)

    # 1. 数据 + RPP + 模型
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
    print(f"已加载: {ckpt}", flush=True)

    # 2. 取一个样本加密
    batch_imgs = next(iter(test_loader)).to(device)
    target = build_target_grid(batch_imgs, device)
    U_cipher = encrypt_batch(
        batch_imgs, CONFIG["l_auth"], rpp_system,
        CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device
    )

    # 3. 验证纯相位 SLM 加载的解密 PSNR
    with torch.no_grad():
        # 方案 A: 完整复振幅 (forward 开头会相位化, 等价于纯相位)
        pred_full = model(U_cipher)
        psnr_full = calculate_psnr(pred_full, target).item()

        # 方案 B: 显式纯相位 (与方案 A 应相同, 因为 forward 开头相位化)
        U_phase_only = torch.exp(1j * torch.angle(U_cipher))
        pred_phase = model(U_phase_only)
        psnr_phase = calculate_psnr(pred_phase, target).item()

    print()
    print("=" * 60)
    print("纯相位 SLM 加载验证")
    print("=" * 60)
    print(f"方案 A (完整复振幅, forward 内部相位化): {psnr_full:.2f} dB")
    print(f"方案 B (显式纯相位 SLM 加载):           {psnr_phase:.2f} dB")
    print(f"两者差异: {abs(psnr_full - psnr_phase):.4f} dB (应≈0, 因 forward 开头相位化)")
    print()

    # 4. 生成 SLM 加载图
    # SLM 加载内容: arg(U_cipher) ∈ [-π, π] -> 映射到 [0, 255]
    size = CONFIG["size"]  # 1080
    phase = torch.angle(U_cipher[0]).cpu().numpy()  # (size, size)
    gray_size = np.clip((phase + np.pi) / (2 * np.pi) * 255 + 0.5, 0, 255).astype(np.uint8)

    # 1080×1080 全息图填入 1920×1080 SLM (高度刚好, 水平居中)
    canvas = np.zeros((SLM_CONFIG["height"], SLM_CONFIG["width"]), dtype=np.uint8)
    y0 = 0  # 高度刚好用满 1080
    x0 = (SLM_CONFIG["width"] - size) // 2  # 水平居中 (1920-1080)/2 = 420
    canvas[y0:y0+size, x0:x0+size] = gray_size

    # 保存
    path_size = os.path.join(SLM_CONFIG["output_dir"], f"slm_phase_{size}x{size}.png")
    path_slm = os.path.join(SLM_CONFIG["output_dir"], f"slm_phase_{SLM_CONFIG['width']}x{SLM_CONFIG['height']}.png")
    Image.fromarray(gray_size).save(path_size)
    Image.fromarray(canvas).save(path_slm)

    print("输出文件:")
    print(f"  {size}×{size} 相位图 (仿真用): {path_size}")
    print(f"  {SLM_CONFIG['width']}×{SLM_CONFIG['height']} SLM 加载图:    {path_slm}")
    print(f"  全息图位置: x=[{x0}, {x0+size}], y=[{y0}, {y0+size}]")
    print()

    # 5. 统计
    print("SLM 加载图统计:")
    print(f"  相位范围: [{phase.min():.4f}, {phase.max():.4f}] rad")
    print(f"  灰度范围: [{gray_size.min()}, {gray_size.max()}] (uint8)")
    print(f"  灰度均值: {gray_size.mean():.2f}, std: {gray_size.std():.2f}")
    print(f"  SLM 全图: {canvas.shape}, 1080×1080 全息图水平居中放置")

    # 6. 可视化
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 行 1: 加密流程
    # 密文振幅
    amp = torch.abs(U_cipher[0]).cpu().numpy()
    im0 = axes[0, 0].imshow(amp, cmap='gray')
    axes[0, 0].set_title(f"Cipher |U| (复振幅振幅)\n范围 [{amp.min():.3f}, {amp.max():.3f}]\n"
                         f"(SLM 丢弃, 不加载)")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    # 密文相位 (= SLM 加载内容)
    im1 = axes[0, 1].imshow(phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[0, 1].set_title(f"Cipher arg(U) = SLM 加载内容\n"
                         f"范围 [-π, π] rad\n"
                         f"(纯相位 SLM 唯一加载的物理量)")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # SLM 加载的 8-bit 灰度图
    im2 = axes[0, 2].imshow(gray_size, cmap='gray', vmin=0, vmax=255)
    axes[0, 2].set_title(f"SLM 加载灰度图 ({size}×{size})\n"
                         f"gray = (phase+π)/(2π)×255\n"
                         f"范围 [0, 255] uint8")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    # 行 2: 解密结果 + SLM 全图
    # 解密结果
    pred_np = pred_phase[0].clamp(0, 1).cpu().numpy()
    axes[1, 0].imshow(pred_np, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title(f"纯相位 SLM 解密结果\n"
                         f"PSNR = {psnr_phase:.2f} dB\n"
                         f"({size}×{size}, 4 通道拼图)")

    # 4 个象限放大
    half = size // 2
    quads = [
        pred_np[0:half, 0:half], pred_np[0:half, half:size],
        pred_np[half:size, 0:half], pred_np[half:size, half:size]
    ]
    combined = np.concatenate([
        np.concatenate([quads[0], quads[1]], axis=1),
        np.concatenate([quads[2], quads[3]], axis=1)
    ], axis=0)
    axes[1, 1].imshow(combined, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title(f"解密 4 通道 (放大)\n"
                         f"左上=Ch0, 右上=Ch1\n左下=Ch2, 右下=Ch3")

    # SLM 全屏图
    axes[1, 2].imshow(canvas, cmap='gray', vmin=0, vmax=255)
    axes[1, 2].set_title(f"SLM 全屏加载图 ({SLM_CONFIG['width']}×{SLM_CONFIG['height']})\n"
                         f"全息图水平居中, 高度占满")
    rect = plt.Rectangle((x0, y0), size, size, linewidth=2,
                         edgecolor='r', facecolor='none')
    axes[1, 2].add_patch(rect)
    axes[1, 2].text(x0 + size // 2, y0 - 30, "Hologram", color='r',
                    ha='center', fontsize=10)

    plt.tight_layout()
    overview_path = os.path.join(SLM_CONFIG["output_dir"], "slm_overview.png")
    plt.savefig(overview_path, dpi=120, bbox_inches='tight')
    print(f"  可视化概览: {overview_path}")

    print()
    print("=" * 60)
    print("SLM 加载说明")
    print("=" * 60)
    print(f"1. 将 slm_phase_1920x1080.png 加载到 Holoeye PLUTO 控制软件")
    print(f"2. SLM 工作波长设为 633 nm (He-Ne 激光)")
    print(f"3. SLM 出射光场 = exp(i·arg(U_cipher)) (纯相位)")
    print(f"4. 后续接入解密光路 (RPP 去除 -> OAM 解复用 -> 传播 -> D2NN)")
    print(f"5. 解密 PSNR = {psnr_phase:.2f} dB (> 30 dB 目标)")


if __name__ == "__main__":
    main()
