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
from font_config import setup_cjk
setup_cjk()
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
    "l_auth": [-15, -5, 5, 15],   # 与 1080 训练一致 (增大 OAM 差异)
    "l_wrong": [-13, -9, 9, 13],
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
    train_dataset = MNISTQuadDataset(mnist_train, img_size=size // 4)  # 小区域增强OAM正交性
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)

    # 2. RPP + 模型
    rpp_system = generate_rpp(size, device)
    theta_max_rad = np.deg2rad(CONFIG["theta_max_deg"]) if CONFIG.get("theta_max_deg") else None
    model = OAM_Crypt_D2NN(
        size=size, num_layers=CONFIG["num_layers"],
        wavelength=CONFIG["wavelength"], pixel_size=CONFIG["pixel_size"],
        z_layer=CONFIG["z_layer"], z0=CONFIG["z0"], rpp=rpp_system,
        oam_keys=CONFIG["l_auth"], z_list=CONFIG["z_list"],
        obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max_rad
    ).to(device)
    optimizer = optim.Adam(model.refine.parameters(), lr=CONFIG["lr"])
    criterion = nn.MSELoss()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 3. 训练 60 epoch (纯 MSE, 图像 size//4 增强OAM正交性)
    print(f"\n[训练] 60 epoch (纯 MSE, 图像区域 {size//4}x{size//4}, OAM={CONFIG['l_auth']})")
    for epoch in range(60):
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
                obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max_rad
            )
            optimizer.zero_grad()
            pred = model(cipher)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_imgs.shape[0]
            n += batch_imgs.shape[0]
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                tb = next(iter(train_loader)).to(device)
                tgt = build_target_grid(tb, device, size=size)
                cp = encrypt_batch(tb, CONFIG["l_auth"], rpp_system,
                    CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
                    size=size, z_list=CONFIG["z_list"], obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max_rad)
                pp = model(cp)
                psnr = calculate_psnr(pp, tgt).item()
            model.train()
            print(f"  Epoch {epoch+1}/60: loss = {epoch_loss/n:.6f}, PSNR = {psnr:.2f} dB")
        else:
            print(f"  Epoch {epoch+1}/60: loss = {epoch_loss/n:.6f}")

    # 4. 验证多平面物理特性 (纯物理, 不用 U-Net)
    print("\n[物理验证] 多平面 OAM 复用全息特性 (纯物理光路, 无 U-Net)")
    print("  使用 sqrt(P) 振幅编码 + 双相位编码 (纯相位 SLM 兼容)")
    print("  4 个图像都在中心同一位置 (多平面复用, 不分象限)")
    model.eval()

    with torch.no_grad():
        batch_imgs = next(iter(train_loader)).to(device)
        # 正确密文
        cipher_correct = encrypt_batch(
            batch_imgs, CONFIG["l_auth"], rpp_system,
            CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
            size=size, z_list=CONFIG["z_list"],
            obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max_rad
        )
        # 错误密文 (用错误 OAM 加密)
        cipher_wrong = encrypt_batch(
            batch_imgs, CONFIG["l_wrong"], rpp_system,
            CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
            size=size, z_list=CONFIG["z_list"],
            obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max_rad
        )

        # 模拟纯相位 SLM 加载: 先去除 RPP, 再双相位编码 + 低通滤波
        U_correct = cipher_correct[0] * torch.conj(rpp_system)
        U_correct = double_phase_encode(U_correct, device)
        U_correct = lowpass_filter(U_correct, sigma=0.15)

        U_wrong = cipher_wrong[0] * torch.conj(rpp_system)
        U_wrong = double_phase_encode(U_wrong, device)
        U_wrong = lowpass_filter(U_wrong, sigma=0.15)

        # 对 4 个 OAM_j 解调, 在 4 个 z_k 平面采样
        # 4 个图像都在中心同一位置, 用归一化互相关衡量图像可见度
        # (正确平面: |U|² ≈ P_j, 相关性高; 错误平面: 散焦, 相关性低)
        img_size = batch_imgs.shape[-1]
        cy = (size - img_size) // 2
        cx = (size - img_size) // 2
        planes_correct = np.zeros((4, 4))
        planes_wrong = np.zeros((4, 4))

        def norm_corr(a, b):
            """归一化互相关: 衡量 a 与 b 的相似度, 范围 [-1, 1]"""
            a = a.float() - a.float().mean()
            b = b.float() - b.float().mean()
            denom = (a.std() * b.std() + 1e-8)
            return (a * b).mean() / denom

        for j, l in enumerate(CONFIG["l_auth"]):
            oam_conj = torch.conj(generate_oam_phase(size, l, device))
            U_demod_correct = U_correct * oam_conj
            U_demod_wrong = U_wrong * oam_conj
            target_j = batch_imgs[0, j]  # 目标图像 j
            for k, z_k in enumerate(CONFIG["z_list"]):
                U_at_z_c = propagate_asm(U_demod_correct, -z_k, CONFIG["wavelength"], CONFIG["pixel_size"], device, theta_max=theta_max_rad)
                U_at_z_w = propagate_asm(U_demod_wrong, -z_k, CONFIG["wavelength"], CONFIG["pixel_size"], device, theta_max=theta_max_rad)
                # 中心区域的强度 |U|² 与目标图像 P_j 的相关性
                I_c = (U_at_z_c.real[cy:cy+img_size, cx:cx+img_size] ** 2 +
                       U_at_z_c.imag[cy:cy+img_size, cx:cx+img_size] ** 2)
                I_w = (U_at_z_w.real[cy:cy+img_size, cx:cx+img_size] ** 2 +
                       U_at_z_w.imag[cy:cy+img_size, cx:cx+img_size] ** 2)
                planes_correct[j, k] = norm_corr(I_c, target_j).item()
                planes_wrong[j, k] = norm_corr(I_w, target_j).item()

    # 5. 打印结果
    z_labels = [f'z={z:.2f}m' for z in CONFIG["z_list"]]
    print("\n[正确 OAM 密文] 各 (OAM_j 解调, z_k 平面) 与目标图像 j 的相关性:")
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

    print(f"\n[错误 OAM 密文] 各 (OAM_j 解调, z_k 平面) 与目标图像 j 的相关性 (应全低):")
    for j in range(4):
        print(f'l={CONFIG["l_auth"][j]:<13}', end='')
        for k in range(4):
            print(f'{planes_wrong[j,k]:<10.4f}  ', end='  ')
        print()

    # 6. 功能验证 (用相关性: 范围 [-1,1], 越高越相似)
    diag = np.mean(np.diag(planes_correct))
    offdiag = (planes_correct.sum() - np.trace(planes_correct).sum()) / 12
    wrong_mean = planes_wrong.mean()
    contrast = diag / max(offdiag, 1e-9)
    sec_ratio = wrong_mean / max(diag, 1e-9)

    print("\n" + "=" * 60)
    print("功能验证结果:")
    print("=" * 60)
    print(f"\n功能1 (不同平面不同图案):")
    print(f"  对角线 (j==k) 平均相关性 = {diag:.4f}")
    print(f"  非对角线 (j!=k) 平均相关性 = {offdiag:.4f}")
    print(f"  对比度 = {contrast:.2f}x  {'✓' if contrast > 1.3 else '✗'}")

    print(f"\n功能2 (错误 OAM 看不到图像):")
    print(f"  错误密文平均相关性 = {wrong_mean:.4f}")
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
            U_at_z = propagate_asm(U_demod, -z_j, CONFIG["wavelength"], CONFIG["pixel_size"], device, theta_max=theta_max_rad)
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
        U_at_z = propagate_asm(U_demod, -z_j, CONFIG["wavelength"], CONFIG["pixel_size"], device, theta_max=theta_max_rad)
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
