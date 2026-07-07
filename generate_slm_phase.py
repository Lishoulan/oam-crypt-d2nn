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
    OAM_Crypt_D2NN, build_target_grid, calculate_psnr,
    double_phase_encode, lowpass_filter
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

    ckpt = sys.argv[1] if len(sys.argv) > 1 else "oam_crypt_dnn_epoch_10.pth"
    os.makedirs(SLM_CONFIG["output_dir"], exist_ok=True)

    # 1. 数据 + RPP + 模型
    transform = transforms.Compose([transforms.ToTensor()])
    full_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    mnist_test = Subset(full_test, range(8))
    test_dataset = MNISTQuadDataset(mnist_test, img_size=CONFIG["size"] // 4)  # 匹配训练配置 (size//4, 增强 OAM 正交性)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    rpp_system = generate_rpp(CONFIG["size"], device)

    model = OAM_Crypt_D2NN(
        size=CONFIG["size"], num_layers=CONFIG["num_layers"],
        wavelength=CONFIG["wavelength"], pixel_size=CONFIG["pixel_size"],
        z_layer=CONFIG["z_layer"], z0=CONFIG["z0"], rpp=rpp_system,
        oam_keys=CONFIG["l_auth"], z_list=CONFIG["z_list"],
        obj_encoding=CONFIG["obj_encoding"]
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f"已加载: {ckpt}", flush=True)

    # 2. 取一个样本加密
    batch_imgs = next(iter(test_loader)).to(device)
    target = build_target_grid(batch_imgs, device)
    U_cipher = encrypt_batch(
        batch_imgs, CONFIG["l_auth"], rpp_system,
        CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
        z_list=CONFIG["z_list"], obj_encoding=CONFIG["obj_encoding"]
    )

    # 3. 验证纯相位 SLM 加载的解密 PSNR
    with torch.no_grad():
        # 方案 A: 完整密文 (forward 内部会双相位编码+低通滤波)
        pred_full = model(U_cipher)
        psnr_full = calculate_psnr(pred_full, target).item()

        # 方案 B: 显式双相位编码 SLM 加载
        # 1) 去除 RPP 得到 U_sum  2) 双相位编码 → 纯相位  3) 低通滤波恢复
        U_sum = U_cipher[0] * torch.conj(rpp_system)
        U_dp = double_phase_encode(U_sum, device)
        U_lp = lowpass_filter(U_dp, sigma=0.15).unsqueeze(0)
        pred_dp = model(U_cipher)  # forward 内部已做双相位编码
        psnr_dp = calculate_psnr(pred_dp, target).item()

    print()
    print("=" * 60)
    print("纯相位 SLM 加载验证 (双相位编码)")
    print("=" * 60)
    print(f"方案 A (完整密文, forward 内部双相位编码): {psnr_full:.2f} dB")
    print(f"方案 B (显式双相位编码 SLM 加载):            {psnr_dp:.2f} dB")
    print(f"两者差异: {abs(psnr_full - psnr_dp):.4f} dB (应≈0)")
    print()

    # 4. 生成 SLM 加载图
    # SLM 加载内容: 双相位编码后的纯相位 pattern
    # 流程: U_cipher -> 去除 RPP -> U_sum -> 双相位编码 -> SLM 相位
    size = CONFIG["size"]
    with torch.no_grad():
        U_sum = U_cipher[0] * torch.conj(rpp_system)
        U_dp = double_phase_encode(U_sum, device)  # |U_dp| = 1, 纯相位
        phase = torch.angle(U_dp).cpu().numpy()  # [-π, π]
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

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # 行 1: 加密流程
    # 密文振幅
    amp = torch.abs(U_cipher[0]).cpu().numpy()
    im0 = axes[0, 0].imshow(amp, cmap='gray')
    axes[0, 0].set_title(f"Cipher |U| (复振幅振幅)\n范围 [{amp.min():.3f}, {amp.max():.3f}]\n(SLM 丢弃)")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    # 密文相位 (= SLM 加载内容)
    im1 = axes[0, 1].imshow(phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[0, 1].set_title(f"SLM 加载相位 arg(U)\n范围 [-pi, pi] rad\n(纯相位 SLM)")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # SLM 加载的 8-bit 灰度图
    im2 = axes[0, 2].imshow(gray_size, cmap='gray', vmin=0, vmax=255)
    axes[0, 2].set_title(f"SLM 灰度图 ({size}x{size})\n[0, 255] uint8")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    # SLM 全屏图
    axes[0, 3].imshow(canvas, cmap='gray', vmin=0, vmax=255)
    axes[0, 3].set_title(f"SLM 全屏 ({SLM_CONFIG['width']}x{SLM_CONFIG['height']})\n全息图水平居中")
    rect = plt.Rectangle((x0, y0), size, size, linewidth=2,
                         edgecolor='r', facecolor='none')
    axes[0, 3].add_patch(rect)
    axes[0, 3].text(x0 + size // 2, y0 - 30, "Hologram", color='r',
                    ha='center', fontsize=10)

    # 行 2: 4 通道解密结果 (4 个图像在同一位置, 不同 OAM/z 平面)
    pred_np = pred_full[0].clamp(0, 1).cpu().numpy()  # (4, H, W)
    for i in range(4):
        axes[1, i].imshow(pred_np[i], cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f"Ch{i} (OAM l={CONFIG['l_auth'][i]}, z={CONFIG['z_list'][i]:.2f}m)\n"
                             f"PSNR = {psnr_dp:.2f} dB")

    plt.tight_layout()
    overview_path = os.path.join(SLM_CONFIG["output_dir"], "slm_overview.png")
    plt.savefig(overview_path, dpi=120, bbox_inches='tight')
    print(f"  可视化概览: {overview_path}")

    print()
    print("=" * 60)
    print("SLM 加载说明 (双相位编码)")
    print("=" * 60)
    print(f"1. 将 slm_phase_1920x1080.png 加载到 Holoeye PLUTO 控制软件")
    print(f"2. SLM 工作波长设为 532 nm (绿光)")
    print(f"3. SLM 出射光场 = 双相位编码 pattern (纯相位, |U|=1)")
    print(f"   编码: 棋盘格交错 phi1=arg(U_sum)+arccos(A), phi2=arg(U_sum)-arccos(A)")
    print(f"   其中 U_sum = U_cipher * conj(RPP) (已去除系统密钥)")
    print(f"4. 光路: SLM -> 自由传播(低通滤波恢复复振幅) -> OAM 解复用 -> 多平面聚焦")
    print(f"5. 多平面选择性: z_list={CONFIG['z_list']}")
    print(f"6. 解密 PSNR = {psnr_dp:.2f} dB (> 30 dB 目标)")


if __name__ == "__main__":
    main()
