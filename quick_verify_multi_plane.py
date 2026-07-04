# -*- coding: utf-8 -*-
"""
快速验证多平面 OAM 复用全息 (振幅编码 + 双相位编码)
====================================================
用 128x128 小尺寸 + 2 epoch 训练, 验证:
  功能1: 不同平面出现不同图案 (对角线能量高)
  功能2: 错误 OAM 看不到图像 (能量低)
  功能3: 正确 OAM 只看到对应平面 (对角线 > 非对角线)

关键: 使用 sqrt(P) 振幅编码 + 双相位编码, 兼容纯相位 SLM 同时有平面选择性。
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

# 临时覆盖 CONFIG (用 128x128 快速验证)
import oam_crypt_d2nn
oam_crypt_d2nn.CONFIG.update({
    "size": 128,
    "batch_size": 8,
    "epochs": 2,
    "warmup_epochs": 0,
    "mid_ch": 32,
    "num_layers": 0,
    "obj_encoding": "amplitude",
})
from oam_crypt_d2nn import (
    CONFIG, generate_rpp, generate_oam_phase, encrypt_batch,
    propagate_asm, MNISTQuadDataset, OAM_Crypt_D2NN,
    build_target_grid, calculate_psnr,
    double_phase_encode, lowpass_filter
)


def main():
    device = torch.device(CONFIG["device"])
    torch.manual_seed(42)
    np.random.seed(42)
    size = CONFIG["size"]

    print(f"快速验证模式: {size}x{size}, z_list={CONFIG['z_list']}")
    print(f"l_auth={CONFIG['l_auth']}")
    print(f"物光编码: {CONFIG['obj_encoding']} (sqrt(P) 振幅编码 + 双相位编码)")

    # 1. 数据
    transform = transforms.Compose([transforms.ToTensor()])
    full_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_train = Subset(full_train, range(64))
    train_dataset = MNISTQuadDataset(mnist_train, img_size=size // 2)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)

    # 2. RPP + 模型
    rpp_system = generate_rpp(size, device)
    model = OAM_Crypt_D2NN(
        size=size, num_layers=CONFIG["num_layers"],
        wavelength=CONFIG["wavelength"], pixel_size=CONFIG["pixel_size"],
        z_layer=CONFIG["z_layer"], z0=CONFIG["z0"], rpp=rpp_system,
        oam_keys=CONFIG["l_auth"], z_list=CONFIG["z_list"],
        obj_encoding=CONFIG["obj_encoding"]
    ).to(device)
    optimizer = optim.Adam(model.refine.parameters(), lr=CONFIG["lr"])
    criterion = nn.MSELoss()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 3. 快速训练 2 epoch
    print("\n[训练] 2 epoch 快速验证...")
    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_loss = 0.0
        n = 0
        for batch_imgs in train_loader:
            batch_imgs = batch_imgs.to(device)
            target = build_target_grid(batch_imgs, device, size=size)
            cipher = encrypt_batch(
                batch_imgs, CONFIG["l_auth"], rpp_system,
                CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
                size=size, z_list=CONFIG["z_list"],
                obj_encoding=CONFIG["obj_encoding"]
            )
            optimizer.zero_grad()
            pred = model(cipher)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_imgs.shape[0]
            n += batch_imgs.shape[0]
        print(f"  Epoch {epoch+1}/{CONFIG['epochs']}: loss = {epoch_loss/n:.6f}")

    # 4. 验证多平面物理特性 (纯物理, 不用 U-Net)
    print("\n[物理验证] 多平面 OAM 复用全息特性 (纯物理光路, 无 U-Net)")
    print("  使用 sqrt(P) 振幅编码 + 双相位编码 (纯相位 SLM 兼容)")
    model.eval()
    half = size // 2
    quad_pos = [(0, 0), (0, half), (half, 0), (half, half)]

    with torch.no_grad():
        batch_imgs = next(iter(train_loader)).to(device)
        # 正确密文
        cipher_correct = encrypt_batch(
            batch_imgs, CONFIG["l_auth"], rpp_system,
            CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
            size=size, z_list=CONFIG["z_list"],
            obj_encoding=CONFIG["obj_encoding"]
        )
        # 错误密文 (用错误 OAM 加密)
        cipher_wrong = encrypt_batch(
            batch_imgs, CONFIG["l_wrong"], rpp_system,
            CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
            size=size, z_list=CONFIG["z_list"],
            obj_encoding=CONFIG["obj_encoding"]
        )

        # 模拟纯相位 SLM 加载: 先去除 RPP, 再双相位编码 + 低通滤波
        # (RPP 必须在双相位编码之前去除, 否则低通滤波破坏 RPP 高频)
        U_correct = cipher_correct[0] * torch.conj(rpp_system)  # 去除 RPP
        U_correct = double_phase_encode(U_correct, device)      # 双相位编码
        U_correct = lowpass_filter(U_correct, sigma=0.15)        # 低通滤波恢复

        U_wrong = cipher_wrong[0] * torch.conj(rpp_system)
        U_wrong = double_phase_encode(U_wrong, device)
        U_wrong = lowpass_filter(U_wrong, sigma=0.15)

        # 对 4 个 OAM_j 解调, 在 4 个 z_k 平面采样
        planes_correct = np.zeros((4, 4))
        planes_wrong = np.zeros((4, 4))

        for j, l in enumerate(CONFIG["l_auth"]):
            oam_conj = torch.conj(generate_oam_phase(size, l, device))
            U_demod_correct = U_correct * oam_conj
            U_demod_wrong = U_wrong * oam_conj
            for k, z_k in enumerate(CONFIG["z_list"]):
                U_at_z_c = propagate_asm(U_demod_correct, -z_k, CONFIG["wavelength"], CONFIG["pixel_size"], device)
                U_at_z_w = propagate_asm(U_demod_wrong, -z_k, CONFIG["wavelength"], CONFIG["pixel_size"], device)
                # 取 j 象限的能量 (加密时 j 图放在 j 象限, 解调后也在 j 象限聚焦)
                qy_j, qx_j = quad_pos[j]
                energy_c = (U_at_z_c.real[qy_j:qy_j+half, qx_j:qx_j+half] ** 2 +
                            U_at_z_c.imag[qy_j:qy_j+half, qx_j:qx_j+half] ** 2).mean().item()
                energy_w = (U_at_z_w.real[qy_j:qy_j+half, qx_j:qx_j+half] ** 2 +
                            U_at_z_w.imag[qy_j:qy_j+half, qx_j:qx_j+half] ** 2).mean().item()
                planes_correct[j, k] = energy_c
                planes_wrong[j, k] = energy_w

    # 5. 打印结果
    z_labels = [f'z={z:.2f}m' for z in CONFIG["z_list"]]
    print("\n[正确 OAM 密文] 各 (OAM_j, z_k) 在 j 象限的能量:")
    print(f"{'OAM_j \\ z_k':<15}", end='')
    for k in range(4):
        print(f'{z_labels[k]:<14}', end='')
    print()
    for j in range(4):
        print(f'l={CONFIG["l_auth"][j]:<13}', end='')
        for k in range(4):
            mark = ' *' if j == k else '  '
            print(f'{planes_correct[j,k]:<10.4f}{mark}', end='  ')
        print('  <- 对角线应高')

    print(f"\n[错误 OAM 密文] 各 (OAM_j, z_k) 在 j 象限的能量 (应全低):")
    for j in range(4):
        print(f'l={CONFIG["l_auth"][j]:<13}', end='')
        for k in range(4):
            print(f'{planes_wrong[j,k]:<10.4f}  ', end='  ')
        print()

    # 6. 功能验证
    diag = np.mean(np.diag(planes_correct))
    offdiag = (planes_correct.sum() - np.trace(planes_correct).sum()) / 12
    wrong_mean = planes_wrong.mean()
    contrast = diag / max(offdiag, 1e-9)
    sec_ratio = wrong_mean / max(diag, 1e-9)

    print("\n" + "=" * 60)
    print("功能验证结果:")
    print("=" * 60)
    print(f"\n功能1 (不同平面不同图案):")
    print(f"  对角线 (j==k) 平均能量 = {diag:.4f}")
    print(f"  非对角线 (j!=k) 平均能量 = {offdiag:.4f}")
    print(f"  对比度 = {contrast:.2f}x  {'✓' if contrast > 1.3 else '✗'}")

    print(f"\n功能2 (错误 OAM 看不到图像):")
    print(f"  错误密文平均能量 = {wrong_mean:.4f}")
    print(f"  安全比 = {sec_ratio:.4f}  {'✓' if sec_ratio < 0.8 else '✗'}")

    print(f"\n功能3 (正确 OAM 只看对应平面):")
    print(f"  对比度 = {contrast:.2f}x  {'✓' if contrast > 1.3 else '✗'}")

    # 7. 可视化
    print("\n[可视化] multi_plane_quick_verify.png")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(
        f'多平面 OAM 复用全息快速验证 (128x128, 2 epoch, 振幅编码+双相位)\n'
        f'功能1: 不同平面不同图 | 功能2: 错误OAM看不到 | 功能3: 正确OAM只看对应平面\n'
        f'对比度={contrast:.2f}x, 安全比={sec_ratio:.4f}',
        fontsize=12, fontweight='bold'
    )

    axes[0, 0].text(0.5, 0.5, '正确 OAM 密文\n用 l_j 解调, 在 z_j 平面采样\n(应看到对应图像)',
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    transform=axes[0, 0].transAxes)
    axes[0, 0].axis('off')

    with torch.no_grad():
        for j, l in enumerate(CONFIG["l_auth"]):
            oam_conj = torch.conj(generate_oam_phase(size, l, device))
            U_demod = U_correct * oam_conj
            z_j = CONFIG["z_list"][j]
            U_at_z = propagate_asm(U_demod, -z_j, CONFIG["wavelength"], CONFIG["pixel_size"], device)
            intensity = (U_at_z.real ** 2 + U_at_z.imag ** 2).cpu().numpy()
            axes[0, j + 1].imshow(intensity, cmap='hot')
            axes[0, j + 1].set_title(f'OAM l={l} 解调\n在 z={z_j:.2f}m 平面\n(看到 l={l} 图像)',
                                      fontsize=9, color='green', fontweight='bold')
            axes[0, j + 1].axis('off')

    axes[1, 0].text(0.5, 0.5, '错误 OAM 密文\n用 l_j 解调, 在 z_j 平面采样\n(应全噪声)',
                    ha='center', va='center', fontsize=10, fontweight='bold', color='red',
                    transform=axes[1, 0].transAxes)
    axes[1, 0].axis('off')

    for j, l in enumerate(CONFIG["l_auth"]):
        oam_conj = torch.conj(generate_oam_phase(size, l, device))
        U_demod = U_wrong * oam_conj
        z_j = CONFIG["z_list"][j]
        U_at_z = propagate_asm(U_demod, -z_j, CONFIG["wavelength"], CONFIG["pixel_size"], device)
        intensity = (U_at_z.real ** 2 + U_at_z.imag ** 2).cpu().numpy()
        axes[1, j + 1].imshow(intensity, cmap='hot')
        axes[1, j + 1].set_title(f'OAM l={l} 解调\n在 z={z_j:.2f}m 平面\n(看不到图像)',
                                  fontsize=9, color='red', fontweight='bold')
        axes[1, j + 1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('multi_plane_quick_verify.png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    print("已保存: multi_plane_quick_verify.png")

    print(f"\n{'='*60}")
    print(f"结论: 对比度 {contrast:.2f}x, 安全比 {sec_ratio:.4f}")
    if contrast > 1.3 and sec_ratio < 0.8:
        print("✓ 多平面 OAM 复用全息 (振幅编码+双相位) 验证通过!")
    else:
        print("⚠ 验证未完全通过, 可能需要更多训练或调整 z_list 间距")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
