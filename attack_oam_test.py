# -*- coding: utf-8 -*-
"""
OAM 涡旋光攻击测试 (改进版)
============================
场景: 攻击者拿到 SLM 全息图 + RPP 系统密钥 (但不知道 OAM 用户密钥)
穷举 l_test ∈ [-10, 10] 尝试解调, 验证非授权 OAM 是否能看到图像。

两层安全性评估:
  1. 纯物理光路 (无 U-Net): OAM 解调 -> ASM(-z) -> 强度采样
  2. U-Net 增强后 (训练好的网络): 用 forward 的 OAM 解复用 + U-Net 精修
     (此时用授权 OAM 解复用, 但输入是用错误 OAM 加密的密文)

度量:
  - 归一化互相关 (形状相似度)
  - PSNR (像素级重建质量, >15 dB 表示能看到图像)
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

import oam_crypt_d2nn
from oam_crypt_d2nn import (
    CONFIG, generate_rpp, generate_oam_phase, encrypt_batch,
    propagate_asm, MNISTQuadDataset, OAM_Crypt_D2NN, build_target_grid,
    calculate_psnr, double_phase_encode, lowpass_filter
)

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def norm_corr(a, b):
    """归一化互相关: 范围 [-1, 1]"""
    a = a.float() - a.float().mean()
    b = b.float() - b.float().mean()
    return (a * b).mean() / (a.std() * b.std() + 1e-8)


def main():
    device = torch.device(CONFIG["device"])
    torch.manual_seed(42)
    np.random.seed(42)
    size = CONFIG["size"]

    print(f"OAM 涡旋光攻击测试 ({size}x{size})")
    print(f"授权 OAM 密钥: {CONFIG['l_auth']}")
    print(f"多平面 z_list: {CONFIG['z_list']} m")

    # 1. 数据 + RPP + 模型
    transform = transforms.Compose([transforms.ToTensor()])
    full_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    mnist_test = Subset(full_test, range(8))
    test_dataset = MNISTQuadDataset(mnist_test, img_size=size // 4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    rpp_system = generate_rpp(size, device)

    model = OAM_Crypt_D2NN(
        size=size, num_layers=CONFIG["num_layers"],
        wavelength=CONFIG["wavelength"], pixel_size=CONFIG["pixel_size"],
        z_layer=CONFIG["z_layer"], z0=CONFIG["z0"], rpp=rpp_system,
        oam_keys=CONFIG["l_auth"], z_list=CONFIG["z_list"],
        obj_encoding=CONFIG["obj_encoding"]
    ).to(device)
    model.load_state_dict(torch.load("oam_crypt_dnn_epoch_10.pth", map_location=device))
    model.eval()
    print("已加载: oam_crypt_dnn_epoch_10.pth")

    batch_imgs = next(iter(test_loader)).to(device)
    img_size = batch_imgs.shape[-1]
    cy = (size - img_size) // 2
    cx = (size - img_size) // 2
    print(f"测试图像尺寸: {img_size}x{img_size}")

    # 2. 授权密文 + SLM 加载模拟
    cipher_auth = encrypt_batch(
        batch_imgs, CONFIG["l_auth"], rpp_system,
        CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
        size=size, z_list=CONFIG["z_list"], obj_encoding=CONFIG["obj_encoding"]
    )

    with torch.no_grad():
        U_sum = cipher_auth[0] * torch.conj(rpp_system)
        U_dp = double_phase_encode(U_sum, device)
        U_slm = lowpass_filter(U_dp, sigma=0.15)

    # ============================================
    # 测试 1: 纯物理光路 - 穷举 OAM 解调
    # ============================================
    print(f"\n{'='*70}")
    print("测试 1: 纯物理光路 (无 U-Net) 穷举 OAM 攻击")
    print(f"{'='*70}")

    l_test_list = list(range(-20, 21))  # 扩大范围到 -20..20 (l_auth=±15)
    z_list = CONFIG["z_list"]
    n_target = 4

    # 相关性张量: (l_test, z_k, target_j)
    corr_matrix = np.zeros((len(l_test_list), len(z_list), n_target))
    psnr_matrix = np.zeros((len(l_test_list), len(z_list), n_target))

    with torch.no_grad():
        for li, l_test in enumerate(l_test_list):
            oam_conj = torch.conj(generate_oam_phase(size, l_test, device))
            U_demod = U_slm * oam_conj
            for k, z_k in enumerate(z_list):
                U_at_z = propagate_asm(U_demod, -z_k, CONFIG["wavelength"], CONFIG["pixel_size"], device)
                I = (U_at_z.real[cy:cy+img_size, cx:cx+img_size] ** 2 +
                     U_at_z.imag[cy:cy+img_size, cx:cx+img_size] ** 2)
                I_norm = (I / (I.max() + 1e-8)).clamp(0, 1)
                for j in range(n_target):
                    target_j = batch_imgs[0, j]
                    corr_matrix[li, k, j] = norm_corr(I, target_j).item()
                    psnr_matrix[li, k, j] = calculate_psnr(I_norm.unsqueeze(0).unsqueeze(0),
                                                            target_j.unsqueeze(0).unsqueeze(0)).item()

    # 对角线 (授权 OAM + 对应 z 平面) vs 非对角线
    diag_corr = [corr_matrix[l_test_list.index(CONFIG["l_auth"][j]), j, j] for j in range(n_target)]
    offdiag_corr = []
    for li, l_test in enumerate(l_test_list):
        for k in range(4):
            for j in range(4):
                if not (l_test in CONFIG["l_auth"] and CONFIG["l_auth"].index(l_test) == k == j):
                    offdiag_corr.append(corr_matrix[li, k, j])

    diag_psnr = [psnr_matrix[l_test_list.index(CONFIG["l_auth"][j]), j, j] for j in range(n_target)]
    offdiag_psnr = []
    for li, l_test in enumerate(l_test_list):
        for k in range(4):
            for j in range(4):
                if not (l_test in CONFIG["l_auth"] and CONFIG["l_auth"].index(l_test) == k == j):
                    offdiag_psnr.append(psnr_matrix[li, k, j])

    print(f"\n对角线 (授权 OAM + 对应 z):")
    for j in range(n_target):
        print(f"  l={CONFIG['l_auth'][j]:>3}, z={z_list[j]:.2f}m, 目标{j}: 相关={diag_corr[j]:.4f}, PSNR={diag_psnr[j]:.2f} dB")
    print(f"  平均: 相关={np.mean(diag_corr):.4f}, PSNR={np.mean(diag_psnr):.2f} dB")

    print(f"\n非对角线 (其他组合): 平均相关={np.mean(offdiag_corr):.4f}, 平均 PSNR={np.mean(offdiag_psnr):.2f} dB")

    auth_max_corr = max(diag_corr)
    unauth_max_corr = max(offdiag_corr)
    contrast = auth_max_corr / max(unauth_max_corr, 1e-9)
    print(f"\n纯物理安全性:")
    print(f"  对角线最大相关 = {auth_max_corr:.4f}")
    print(f"  非对角线最大相关 = {unauth_max_corr:.4f}")
    print(f"  对比度 = {contrast:.2f}x  {'✓' if contrast > 2 else '✗ 物理正交性不足'}")

    # ============================================
    # 测试 2: U-Net 增强后 - 错误 OAM 加密攻击
    # ============================================
    print(f"\n{'='*70}")
    print("测试 2: U-Net 增强后 (用错误 OAM 加密, 正确 OAM 解密)")
    print(f"{'='*70}")

    # 用各种错误 OAM 加密, 然后用授权 OAM 解密
    l_attack_list = [-13, -9, -7, -3, -1, 0, 1, 3, 7, 9, 13]  # 非授权 OAM
    print(f"攻击 OAM: {l_attack_list}")
    print(f"对照: 授权 OAM {CONFIG['l_auth']}\n")

    auth_psnr_list = []
    unauth_psnr_list = []

    with torch.no_grad():
        # 授权加密 -> 授权解密 (对照)
        pred_auth = model(cipher_auth)
        psnr_auth = calculate_psnr(pred_auth, build_target_grid(batch_imgs, device, size=size)).item()
        auth_psnr_list.append(psnr_auth)
        print(f"授权加密 -> 授权解密 (对照): PSNR = {psnr_auth:.2f} dB")

        # 错误 OAM 加密 -> 授权 OAM 解密 (攻击)
        for l_attack in l_attack_list:
            cipher_attack = encrypt_batch(
                batch_imgs, [l_attack] * 4, rpp_system,
                CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
                size=size, z_list=CONFIG["z_list"], obj_encoding=CONFIG["obj_encoding"]
            )
            pred_attack = model(cipher_attack)
            psnr_attack = calculate_psnr(pred_attack, build_target_grid(batch_imgs, device, size=size)).item()
            # 平均能量 (越低越好)
            mean_energy = pred_attack.abs().mean().item()
            unauth_psnr_list.append((l_attack, psnr_attack, mean_energy))
            print(f"  l_attack={l_attack:>3} 加密 -> 授权 OAM 解密: PSNR = {psnr_attack:.2f} dB, 平均能量 = {mean_energy:.4f}")

    print(f"\nU-Net 增强安全性:")
    print(f"  授权 PSNR = {psnr_auth:.2f} dB")
    print(f"  非授权平均 PSNR = {np.mean([p for _, p, _ in unauth_psnr_list]):.2f} dB")
    print(f"  非授权最大 PSNR = {max(p for _, p, _ in unauth_psnr_list):.2f} dB")
    sec_ratio = np.mean([e for _, _, e in unauth_psnr_list]) / pred_auth.abs().mean().item()
    print(f"  安全比 (能量) = {sec_ratio:.4f}  {'✓' if sec_ratio < 0.5 else '✗'}")

    # ============================================
    # 可视化
    # ============================================
    print(f"\n[可视化] 生成攻击测试图...")

    # 图 1: 热力图 (相关性矩阵)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f'OAM 涡旋光攻击测试 - 纯物理光路 ({size}x{size})\n'
        f'授权 OAM = {CONFIG["l_auth"]}, z_list = {z_list}\n'
        f'对角线平均相关 {np.mean(diag_corr):.4f} vs 非对角线平均相关 {np.mean(offdiag_corr):.4f}',
        fontsize=12, fontweight='bold'
    )

    for j in range(n_target):
        ax = axes[j // 2, j % 2]
        mat = corr_matrix[:, :, j].T
        im = ax.imshow(mat, aspect='auto', cmap='RdYlGn', vmin=-0.3, vmax=0.8, origin='lower')
        ax.set_xticks(range(len(l_test_list)))
        ax.set_xticklabels([str(l) for l in l_test_list], rotation=45, fontsize=7)
        ax.set_yticks(range(len(z_list)))
        ax.set_yticklabels([f'z={z:.2f}' for z in z_list])
        ax.set_xlabel('攻击 OAM 拓扑荷 l_test')
        ax.set_ylabel('采样平面 z_k')
        auth_idx = l_test_list.index(CONFIG["l_auth"][j])
        ax.set_title(f'目标 {j} (l={CONFIG["l_auth"][j]}, z={z_list[j]:.2f}m)\n'
                    f'正确点 (l={CONFIG["l_auth"][j]}, z={z_list[j]:.2f}) 对角={diag_corr[j]:.4f}',
                    fontsize=9, color='green', fontweight='bold')
        ax.axvline(x=auth_idx, color='cyan', linewidth=1.5, linestyle='--', alpha=0.7)
        ax.axhline(y=j, color='cyan', linewidth=1.5, linestyle='--', alpha=0.7)
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('attack_oam_heatmap.png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  已保存: attack_oam_heatmap.png")

    # 图 2: 解调图像示例 (5 个 OAM x 4 个 z 平面)
    show_l = [-15, -13, -5, -1, 0, 5, 15]  # 含授权和非授权
    fig2, axes2 = plt.subplots(len(show_l), 4, figsize=(18, 4 * len(show_l)))
    fig2.suptitle(
        f'OAM 攻击解密图像示例 (中心 {img_size}x{img_size})\n'
        f'行=攻击 OAM, 列=采样 z 平面\n'
        f'授权 OAM (l={CONFIG["l_auth"]}) 在对角线应有清晰图像, 其他为噪声',
        fontsize=12, fontweight='bold'
    )

    with torch.no_grad():
        for row, l_test in enumerate(show_l):
            oam_conj = torch.conj(generate_oam_phase(size, l_test, device))
            U_demod = U_slm * oam_conj
            for col, z_k in enumerate(z_list):
                U_at_z = propagate_asm(U_demod, -z_k, CONFIG["wavelength"], CONFIG["pixel_size"], device)
                I = (U_at_z.real[cy:cy+img_size, cx:cx+img_size] ** 2 +
                     U_at_z.imag[cy:cy+img_size, cx:cx+img_size] ** 2).cpu().numpy()
                axes2[row, col].imshow(I, cmap='hot')
                is_auth = l_test in CONFIG["l_auth"]
                tag = "授权" if is_auth else "非授权"
                color = 'green' if is_auth else 'red'
                match = (CONFIG["l_auth"].index(l_test) == col) if is_auth else False
                mark = "  ✓" if match else ""
                li = l_test_list.index(l_test)
                c_val = corr_matrix[li, col, col] if col < n_target else 0
                axes2[row, col].set_title(
                    f'l={l_test} ({tag})\nz={z_k:.2f}m\n相关={c_val:.4f}{mark}',
                    fontsize=8, color=color, fontweight='bold'
                )
                axes2[row, col].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('attack_oam_images.png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  已保存: attack_oam_images.png")

    # ============================================
    # 总结
    # ============================================
    print(f"\n{'='*70}")
    print("攻击测试总结")
    print(f"{'='*70}")
    print(f"\n1. 纯物理光路 (无 U-Net):")
    print(f"   对角线平均相关 = {np.mean(diag_corr):.4f}, PSNR = {np.mean(diag_psnr):.2f} dB")
    print(f"   非对角线平均相关 = {np.mean(offdiag_corr):.4f}, PSNR = {np.mean(offdiag_psnr):.2f} dB")
    print(f"   对比度 = {contrast:.2f}x")
    print(f"\n2. U-Net 增强后:")
    print(f"   授权 PSNR = {psnr_auth:.2f} dB")
    print(f"   非授权平均 PSNR = {np.mean([p for _, p, _ in unauth_psnr_list]):.2f} dB")
    print(f"   安全比 = {sec_ratio:.4f}")
    print(f"\n结论:")
    if contrast > 2 and sec_ratio < 0.5:
        print("  ✓ 系统抗 OAM 攻击能力强")
    elif sec_ratio < 0.5:
        print(f"  ⚠ 物理正交性不足 (对比 {contrast:.2f}x), 但 U-Net 增强后安全 (安全比 {sec_ratio:.4f})")
        print(f"    说明: 安全性主要依赖 U-Net 学习的映射, 而非物理 OAM 正交性")
    else:
        print(f"  ✗ 系统安全性不足, 需要增大 OAM 拓扑荷差异或缩小图像区域")


if __name__ == "__main__":
    main()
