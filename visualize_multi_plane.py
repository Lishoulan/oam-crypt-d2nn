# -*- coding: utf-8 -*-
"""
多平面 OAM 复用全息可视化
==========================
演示 3 个核心功能:
  1. 不同平面出现不同图案 (每路 OAM 对应一个独立 z 平面)
  2. 错误 OAM 密钥看不到图像 (任何平面都看不到)
  3. 正确 OAM 密钥只能看到对应平面的图案 (其他平面散焦)

用法: py visualize_multi_plane.py [checkpoint.pth]
"""
import os
import sys
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from oam_crypt_d2nn import (
    CONFIG, generate_rpp, generate_oam_phase, encrypt_batch,
    propagate_asm, MNISTQuadDataset, OAM_Crypt_D2NN
)


def main():
    device = torch.device(CONFIG["device"])
    torch.manual_seed(42)
    np.random.seed(42)

    ckpt = sys.argv[1] if len(sys.argv) > 1 else "oam_crypt_dnn_epoch_20.pth"
    if not os.path.exists(ckpt):
        print(f"ERROR: checkpoint {ckpt} not found. Train first or pass a valid path.")
        sys.exit(1)

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
        oam_keys=CONFIG["l_auth"], z_list=CONFIG["z_list"]
    ).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded: {ckpt}")
    print(f"z_list = {CONFIG['z_list']} m (4 个独立平面)")
    print(f"l_auth = {CONFIG['l_auth']} (4 个 OAM 密钥)")

    # 2. 取一组测试图像
    batch_imgs = next(iter(test_loader)).to(device)  # (1, 4, S, S)
    size = CONFIG["size"]
    half = size // 2

    # 3. 用正确 OAM 加密
    cipher_auth = encrypt_batch(
        batch_imgs, CONFIG["l_auth"], rpp_system,
        CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
        size=size, z_list=CONFIG["z_list"]
    )

    # 4. 用错误 OAM 加密 (模拟"非授权光")
    cipher_wrong = encrypt_batch(
        batch_imgs, CONFIG["l_wrong"], rpp_system,
        CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
        size=size, z_list=CONFIG["z_list"]
    )

    # =========================================================================
    # 功能 1+3: 正确 OAM 解调 -> 在 4 个不同平面采样, 每个平面只看到对应图
    # =========================================================================
    # 物理流程: cipher → ×conj(RPP) → ×conj(OAM_j) → 在 z 平面观察
    # 通道 j 信号: 在 z=z_list[j] 平面聚焦成图像 j, 在其他平面散焦成噪声
    print("\n[功能 1+3] 正确 OAM 解调 -> 在 4 个平面采样...")
    planes_correct = torch.zeros(4, 4, size, size)  # (j_OAM, z_plane, H, W)
    with torch.no_grad():
        U = cipher_auth[0]  # (H, W) complex
        U = torch.exp(1j * torch.angle(U))  # 纯相位 SLM 加载
        U = U * torch.conj(rpp_system)

        for j, l in enumerate(CONFIG["l_auth"]):
            oam_conj = torch.conj(generate_oam_phase(size, l, device))
            U_demod = U * oam_conj
            # 在 4 个不同平面采样
            for k, z_k in enumerate(CONFIG["z_list"]):
                U_at_z = propagate_asm(U_demod, -z_k, CONFIG["wavelength"], CONFIG["pixel_size"], device)
                # 取强度
                intensity = (U_at_z.real ** 2 + U_at_z.imag ** 2)
                planes_correct[j, k] = intensity.cpu()

    # =========================================================================
    # 功能 2: 错误 OAM 解调 -> 任何平面都看不到图像
    # =========================================================================
    print("[功能 2] 错误 OAM 解调 -> 在 4 个平面采样 (应该全噪声)...")
    planes_wrong = torch.zeros(4, 4, size, size)
    with torch.no_grad():
        U = cipher_wrong[0]
        U = torch.exp(1j * torch.angle(U))
        U = U * torch.conj(rpp_system)

        # 用 4 个正确 OAM 解调错误密文 (任何都看不到)
        for j, l in enumerate(CONFIG["l_auth"]):
            oam_conj = torch.conj(generate_oam_phase(size, l, device))
            U_demod = U * oam_conj
            for k, z_k in enumerate(CONFIG["z_list"]):
                U_at_z = propagate_asm(U_demod, -z_k, CONFIG["wavelength"], CONFIG["pixel_size"], device)
                intensity = (U_at_z.real ** 2 + U_at_z.imag ** 2)
                planes_wrong[j, k] = intensity.cpu()

    # =========================================================================
    # 可视化
    # =========================================================================
    print("\n[可视化] 生成 multi_plane_demo.png ...")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(4, 9, figsize=(27, 12))

    fig.suptitle(
        '多平面 OAM 复用全息演示\n'
        f'功能1: 不同平面出现不同图案 | 功能2: 错误 OAM 看不到 | 功能3: 正确 OAM 只看到对应平面\n'
        f'z_list = {CONFIG["z_list"]} m, l_auth = {CONFIG["l_auth"]}',
        fontsize=14, fontweight='bold', y=0.995
    )

    z_labels = [f'z={z:.2f}m' for z in CONFIG["z_list"]]

    # 行 0-3: 每行用一个 OAM_j 解调, 在 4 个平面采样 + 错误密文对比
    for j in range(4):
        l_j = CONFIG["l_auth"][j]
        # 列 0: 标签
        axes[j, 0].text(0.5, 0.5, f'用 OAM l={l_j}\n解调\n\n观察 4 个平面:',
                        ha='center', va='center', fontsize=12, fontweight='bold',
                        transform=axes[j, 0].transAxes)
        axes[j, 0].axis('off')
        if j == 0:
            axes[j, 0].set_title('解调密钥', fontsize=12, fontweight='bold')

        # 列 1-4: 正确密文, 在 4 个平面采样
        for k in range(4):
            ax = axes[j, k + 1]
            # crop 到对应象限放大显示
            quad_y, quad_x = [(0, 0), (0, half), (half, 0), (half, half)][k]
            img_quad = planes_correct[j, k, quad_y:quad_y + half, quad_x:quad_x + half].numpy()
            ax.imshow(img_quad, cmap='gray')
            # 标记对角线 (期望 j == k 时此象限有图像)
            focus = "✓ 聚焦" if j == k else "✗ 散焦"
            color = 'green' if j == k else 'red'
            ax.set_title(f'{z_labels[k]}\n{focus}', color=color, fontsize=10, fontweight='bold')
            ax.axis('off')
            if j == 0:
                pass  # title already set above for col 0

        # 列 5: 分隔
        axes[j, 5].text(0.5, 0.5, 'vs\n错误 OAM\n密文',
                        ha='center', va='center', fontsize=10, color='red',
                        transform=axes[j, 5].transAxes)
        axes[j, 5].axis('off')

        # 列 6-9: 错误密文, 在 4 个平面采样 (应该全噪声)
        for k in range(4):
            ax = axes[j, k + 6]
            quad_y, quad_x = [(0, 0), (0, half), (half, 0), (half, half)][k]
            img_quad = planes_wrong[j, k, quad_y:quad_y + half, quad_x:quad_x + half].numpy()
            ax.imshow(img_quad, cmap='gray')
            ax.set_title(f'{z_labels[k]}\n(噪声)', color='gray', fontsize=10)
            ax.axis('off')
            if j == 0:
                ax.text(0.5, 1.08, '错误 OAM 密文 (任何平面都看不到)',
                        ha='center', va='bottom', fontsize=11, color='red', fontweight='bold',
                        transform=ax.transAxes)

    # 添加列组标题
    for ax_idx, title in [(1, '正确 OAM 密文 - 在 4 个平面采样'), (6, '')]:
        if title:
            axes[0, ax_idx].text(0.5, 1.15, title,
                                  ha='center', va='bottom', fontsize=12, color='green', fontweight='bold',
                                  transform=axes[0, ax_idx].transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('multi_plane_demo.png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()
    print("已保存: multi_plane_demo.png")

    # =========================================================================
    # 输出统计: 验证 3 个功能
    # =========================================================================
    print("\n" + "=" * 70)
    print("功能验证统计")
    print("=" * 70)

    # 计算每个 (j_OAM, z_plane) 组合的能量集中度
    # 期望: 正确密文, j == k 时, 对应象限能量远高于其他象限
    print("\n[正确 OAM 密文] 各 (OAM_j 解调, z_k 平面) 在对应象限的能量:")
    print(f"{'OAM_j \\ z_k':<15}", end='')
    for k in range(4):
        print(f'{z_labels[k]:<14}', end='')
    print()
    for j in range(4):
        print(f'l={CONFIG["l_auth"][j]:<13}', end='')
        for k in range(4):
            quad_y, quad_x = [(0, 0), (0, half), (half, 0), (half, half)][k]
            energy = planes_correct[j, k, quad_y:quad_y + half, quad_x:quad_x + half].mean().item()
            mark = ' *' if j == k else '  '
            print(f'{energy:<10.4f}{mark}', end='  ')
        print('  <- 应该 j==k 时高 (对角线高)')

    print("\n[错误 OAM 密文] 各 (OAM_j 解调, z_k 平面) 在对应象限的能量 (应该全部低):")
    print(f"{'OAM_j \\ z_k':<15}", end='')
    for k in range(4):
        print(f'{z_labels[k]:<14}', end='')
    print()
    for j in range(4):
        print(f'l={CONFIG["l_auth"][j]:<13}', end='')
        for k in range(4):
            quad_y, quad_x = [(0, 0), (0, half), (half, 0), (half, half)][k]
            energy = planes_wrong[j, k, quad_y:quad_y + half, quad_x:quad_x + half].mean().item()
            print(f'{energy:<10.4f}  ', end='  ')
        print()

    # =========================================================================
    # 简明结论
    # =========================================================================
    print("\n" + "=" * 70)
    print("结论:")
    print("=" * 70)
    diag_correct = [planes_correct[j, j, *[(0, 0), (0, half), (half, 0), (half, half)][j]].mean().item() for j in range(4)]
    offdiag_correct = []
    for j in range(4):
        for k in range(4):
            if j != k:
                qy, qx = [(0, 0), (0, half), (half, 0), (half, half)][k]
                offdiag_correct.append(planes_correct[j, k, qy:qy + half, qx:qx + half].mean().item())
    wrong_mean = planes_wrong.mean().item()
    diag_mean = float(np.mean(diag_correct))
    offdiag_mean = float(np.mean(offdiag_correct))

    print(f"\n功能1 (不同平面不同图案):")
    print(f"  对角线 (j==k, 正确平面) 平均能量 = {diag_mean:.4f}")
    print(f"  非对角线 (j!=k, 错误平面) 平均能量 = {offdiag_mean:.4f}")
    print(f"  对比度 = {diag_mean / max(offdiag_mean, 1e-9):.2f}x  (越高越好)")
    print(f"  {'✓ 通过' if diag_mean > 1.5 * offdiag_mean else '✗ 未通过'}")

    print(f"\n功能2 (错误 OAM 看不到图像):")
    print(f"  错误密文所有平面平均能量 = {wrong_mean:.4f}")
    print(f"  正确密文对角线能量 = {diag_mean:.4f}")
    print(f"  安全比 = {wrong_mean / max(diag_mean, 1e-9):.4f}  (越低越安全)")
    print(f"  {'✓ 通过' if wrong_mean < 0.7 * diag_mean else '✗ 未通过'}")

    print(f"\n功能3 (正确 OAM 只看对应平面):")
    print(f"  对角线 vs 非对角线能量比 = {diag_mean / max(offdiag_mean, 1e-9):.2f}x")
    print(f"  {'✓ 通过' if diag_mean > 1.5 * offdiag_mean else '✗ 未通过'}")


if __name__ == "__main__":
    main()
