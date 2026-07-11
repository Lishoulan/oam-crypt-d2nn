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
    mnist_test = Subset(full_test, range(50))  # 50 样本统计 (训练用 200, 这里折中平衡速度)
    test_dataset = MNISTQuadDataset(mnist_test, img_size=size // 4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    rpp_placeholder = generate_rpp(size, device)  # 占位, 实际从 checkpoint 恢复
    theta_max_rad = np.deg2rad(CONFIG["theta_max_deg"]) if CONFIG.get("theta_max_deg") else None

    model = OAM_Crypt_D2NN(
        size=size, num_layers=CONFIG["num_layers"],
        wavelength=CONFIG["wavelength"], pixel_size=CONFIG["pixel_size"],
        z_layer=CONFIG["z_layer"], z0=CONFIG["z0"], rpp=rpp_placeholder,
        oam_keys=CONFIG["l_auth"], z_list=CONFIG["z_list"],
        obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max_rad
    ).to(device)
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "oam_crypt_dnn_epoch_18.pth"  # 最佳模型 (3层D2NN)
    ckpt_data = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt_data, dict) and 'model_state_dict' in ckpt_data:
        model.load_state_dict(ckpt_data['model_state_dict'])
        print(f"已加载: {ckpt_path} (PSNR={ckpt_data.get('psnr','?'):.2f} dB, SR={ckpt_data.get('sec_ratio','?'):.4f})")
    else:
        model.load_state_dict(ckpt_data)
        print(f"已加载: {ckpt_path} (旧格式)")
    model.eval()

    # 从模型 buffer 取训练时的 RPP (rpp_conj 的共轭 = 原始 rpp)
    rpp_system = torch.conj(model.rpp_conj)
    print(f"已从 checkpoint 恢复训练时的 RPP 系统密钥")

    batch_imgs = next(iter(test_loader)).to(device)
    img_size = batch_imgs.shape[-1]
    cy = (size - img_size) // 2
    cx = (size - img_size) // 2
    print(f"测试图像尺寸: {img_size}x{img_size}")

    # 2. 授权密文 + SLM 加载模拟
    cipher_auth = encrypt_batch(
        batch_imgs, CONFIG["l_auth"], rpp_system,
        CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
        size=size, z_list=CONFIG["z_list"], obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max_rad
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
                U_at_z = propagate_asm(U_demod, -z_k, CONFIG["wavelength"], CONFIG["pixel_size"], device, theta_max=theta_max_rad)
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
    # 测试 2: U-Net 增强后 - 多样本 OAM 攻击 + RPP 攻击
    # ============================================
    print(f"\n{'='*70}")
    print("测试 2: U-Net 增强后 - 多样本统计 (50 样本)")
    print(f"{'='*70}")

    l_attack_list = [-13, -9, -7, -3, -1, 0, 1, 3, 7, 9, 13]  # 全部非授权 OAM
    l_wrong_trained = [-13, -9, -7, -3, 3, 7, 9, 13]  # 训练过的 l_wrong (不含 -1,0,1)
    l_untrained = [-1, 0, 1]  # 未训练过的小拓扑荷

    print(f"攻击 OAM: {l_attack_list}")
    print(f"训练过的: {l_wrong_trained}")
    print(f"未训练过的: {l_untrained}")
    print(f"对照: 授权 OAM {CONFIG['l_auth']}\n")

    # 多样本统计
    auth_energies = []      # 授权解密能量 (abs mean)
    auth_psnrs = []         # 授权 PSNR
    oam_attack_energies = {l: [] for l in l_attack_list}  # 每个错误 OAM 的能量
    rpp_attack_energies = []  # RPP 攻击能量

    n_samples = 0
    with torch.no_grad():
        for si, batch_i in enumerate(test_loader):
            batch_i = batch_i.to(device)
            target = build_target_grid(batch_i, device, size=size)

            # 授权加密 -> 授权解密 (对照)
            cipher_a = encrypt_batch(
                batch_i, CONFIG["l_auth"], rpp_system,
                CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
                size=size, z_list=CONFIG["z_list"], obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max_rad
            )
            pred_a = model(cipher_a)
            auth_energies.append(pred_a.abs().mean().item())
            auth_psnrs.append(calculate_psnr(pred_a, target).item())

            # 错误 OAM 加密 -> 授权解密 (攻击)
            for l_attack in l_attack_list:
                cipher_atk = encrypt_batch(
                    batch_i, [l_attack] * 4, rpp_system,
                    CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
                    size=size, z_list=CONFIG["z_list"], obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max_rad
                )
                pred_atk = model(cipher_atk)
                oam_attack_energies[l_attack].append(pred_atk.abs().mean().item())

            # RPP 攻击: 正确 OAM + 错误 RPP (与训练验证一致!)
            rpp_wrong = generate_rpp(size, device)
            cipher_rpp = encrypt_batch(
                batch_i, CONFIG["l_auth"], rpp_wrong,
                CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
                size=size, z_list=CONFIG["z_list"], obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max_rad
            )
            pred_rpp = model(cipher_rpp)
            rpp_attack_energies.append(pred_rpp.abs().mean().item())

            n_samples += 1
            if si % 10 == 0:
                print(f"  已处理 {si}/{len(test_loader)} 样本...")

    # 汇总统计
    auth_energy_mean = np.mean(auth_energies)
    auth_psnr_mean = np.mean(auth_psnrs)
    print(f"\n--- 结果 ({n_samples} 样本统计) ---")
    print(f"授权解密: 平均能量={auth_energy_mean:.6f}, 平均PSNR={auth_psnr_mean:.2f} dB")

    print(f"\nOAM 攻击 (错误 OAM 加密 + 正确 RPP):")
    oam_energy_means = {}
    for l in l_attack_list:
        oam_energy_means[l] = np.mean(oam_attack_energies[l])
        tag = "(训练过)" if l in l_wrong_trained else "(未训练)"
        print(f"  l={l:>3} {tag}: 平均能量={oam_energy_means[l]:.6f}, SR={oam_energy_means[l]/auth_energy_mean:.4f}")

    print(f"\nRPP 攻击 (正确 OAM + 错误 RPP, 与训练验证一致):")
    rpp_energy_mean = np.mean(rpp_attack_energies)
    sr_rpp = rpp_energy_mean / auth_energy_mean
    print(f"  平均能量={rpp_energy_mean:.6f}, SR={sr_rpp:.4f}")

    # 安全比对比
    sr_all_oam = np.mean([oam_energy_means[l] for l in l_attack_list]) / auth_energy_mean
    sr_trained_oam = np.mean([oam_energy_means[l] for l in l_wrong_trained]) / auth_energy_mean
    sr_untrained_oam = np.mean([oam_energy_means[l] for l in l_untrained]) / auth_energy_mean

    print(f"\n--- 安全比对比 (与训练 SR 0.091 对比) ---")
    print(f"  训练日志 SR (RPP攻击, 200样本, 无abs): 0.091 (epoch_18)")
    print(f"  本次 RPP攻击 SR (50样本, 有abs):      {sr_rpp:.4f}")
    print(f"  本次 OAM攻击 SR 全部11个 (有abs):     {sr_all_oam:.4f}")
    print(f"  本次 OAM攻击 SR 训练过的8个 (有abs):  {sr_trained_oam:.4f}")
    print(f"  本次 OAM攻击 SR 未训练的3个 (有abs):   {sr_untrained_oam:.4f}")

    print(f"\n结论:")
    if sr_rpp < 0.5:
        print(f"  ✓ RPP 攻击安全 (SR={sr_rpp:.4f}) — 与训练一致")
    else:
        print(f"  ✗ RPP 攻击不安全 (SR={sr_rpp:.4f})")
    if sr_trained_oam < 0.5:
        print(f"  ✓ 训练过的 OAM 攻击安全 (SR={sr_trained_oam:.4f})")
    else:
        print(f"  ✗ 训练过的 OAM 攻击不安全 (SR={sr_trained_oam:.4f})")
    if sr_untrained_oam > sr_trained_oam * 1.5:
        print(f"  ⚠ 未训练的 OAM (l=±1,0) 泛化不足: SR={sr_untrained_oam:.4f} >> 训练过的 {sr_trained_oam:.4f}")

    # 用变量保存给后续可视化使用
    psnr_auth = auth_psnr_mean
    sec_ratio = sr_all_oam

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
                U_at_z = propagate_asm(U_demod, -z_k, CONFIG["wavelength"], CONFIG["pixel_size"], device, theta_max=theta_max_rad)
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
