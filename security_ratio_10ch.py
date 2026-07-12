# -*- coding: utf-8 -*-
"""
10 通道 OAM-MDNN SecurityRatio 测试
====================================
对训练好的模型做两类攻击测试:
  1. RPP 攻击: 正确 OAM + 错误 RPP
  2. OAM 攻击: 错误 OAM + 正确 RPP

对每个测试样本:
  - 算合法解密的 PSNR_C (主信道)
  - 算攻击后解密的 PSNR_C
  - SecurityRatio = PSNR_攻击 / PSNR_合法 (越低越好)

指标:
  - 平均 PSNR_C (合法) — 应 > 30 dB
  - 平均 PSNR_C (RPP 攻击) — 应 < 10 dB
  - 平均 PSNR_C (OAM 攻击) — 应 < 10 dB
  - SecurityRatio_RPP — 应 < 0.3
  - SecurityRatio_OAM — 应 < 0.3
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader
import torchvision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oam_crypt_d2nn import (
    CONFIG, OAM_Crypt_D2NN, encrypt_batch, build_target_grid,
    generate_rpp, calculate_center_psnr, security_ratio,
)


def load_model(checkpoint_path, device):
    """加载训练好的 model"""
    rpp = generate_rpp(CONFIG["size"], device, generator=torch.Generator(device).manual_seed(0))
    model = OAM_Crypt_D2NN(
        size=CONFIG["size"], num_layers=CONFIG["num_layers"],
        wavelength=CONFIG["wavelength"], pixel_size=CONFIG["pixel_size"],
        z_layer=CONFIG["z_layer"], z0=CONFIG["z0"], rpp=rpp,
        oam_keys=CONFIG["l_auth"], z_list=CONFIG["z_list"],
        obj_encoding=CONFIG["obj_encoding"],
        theta_max=CONFIG["theta_max_deg"] * np.pi / 180,
        slm_aware=CONFIG["slm_aware"],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model, rpp


def run_attack_test(model, rpp_system, test_loader, device):
    """
    对测试集做攻击测试
    Returns:
        results: dict of lists
    """
    num_channels = len(CONFIG["l_auth"])
    theta_max_rad = CONFIG["theta_max_deg"] * np.pi / 180

    psnr_auth_list = []
    psnr_rpp_list = []
    psnr_oam_list = []
    sr_rpp_list = []
    sr_oam_list = []

    # 错误 OAM 密钥(全部 10 个)
    l_wrong_full = CONFIG["l_wrong"][:num_channels]

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            tgt = build_target_grid(batch, device, size=CONFIG["size"])

            # 1. 合法解密
            c_auth = encrypt_batch(
                batch, CONFIG["l_auth"], rpp_system,
                CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
                size=CONFIG["size"], z_list=CONFIG["z_list"],
                obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max_rad
            )
            p_auth = model(c_auth)

            # 2. RPP 攻击
            rpp_wrong = generate_rpp(CONFIG["size"], device)
            c_rpp = encrypt_batch(
                batch, CONFIG["l_auth"], rpp_wrong,
                CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
                size=CONFIG["size"], z_list=CONFIG["z_list"],
                obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max_rad
            )
            p_rpp = model(c_rpp)

            # 3. OAM 攻击
            c_oam = encrypt_batch(
                batch, l_wrong_full, rpp_system,
                CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
                size=CONFIG["size"], z_list=CONFIG["z_list"],
                obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max_rad
            )
            p_oam = model(c_oam)

            # 4. 算指标
            psnr_auth = calculate_center_psnr(p_auth, tgt).item()
            psnr_rpp = calculate_center_psnr(p_rpp, tgt).item()
            psnr_oam = calculate_center_psnr(p_oam, tgt).item()
            sr_rpp = security_ratio(p_auth, p_rpp).item()
            sr_oam = security_ratio(p_auth, p_oam).item()

            psnr_auth_list.append(psnr_auth)
            psnr_rpp_list.append(psnr_rpp)
            psnr_oam_list.append(psnr_oam)
            sr_rpp_list.append(sr_rpp)
            sr_oam_list.append(sr_oam)

    return {
        "psnr_auth": psnr_auth_list,
        "psnr_rpp": psnr_rpp_list,
        "psnr_oam": psnr_oam_list,
        "sr_rpp": sr_rpp_list,
        "sr_oam": sr_oam_list,
    }


def print_report(results):
    """打印 SecurityRatio 测试报告"""
    print("\n" + "=" * 70)
    print("10 通道 OAM-MDNN SecurityRatio 测试报告")
    print("=" * 70)
    print(f"  测试样本数: {len(results['psnr_auth'])}")
    print("-" * 70)
    print(f"  平均 PSNR_C (合法解密):       {np.mean(results['psnr_auth']):.2f} dB")
    print(f"  平均 PSNR_C (RPP 攻击):       {np.mean(results['psnr_rpp']):.2f} dB")
    print(f"  平均 PSNR_C (OAM 攻击):       {np.mean(results['psnr_oam']):.2f} dB")
    print("-" * 70)
    print(f"  平均 SecurityRatio (RPP 攻击): {np.mean(results['sr_rpp']):.4f}  (目标 < 0.3)")
    print(f"  平均 SecurityRatio (OAM 攻击): {np.mean(results['sr_oam']):.4f}  (目标 < 0.3)")
    print("-" * 70)

    # 评估
    sr_rpp_mean = np.mean(results['sr_rpp'])
    sr_oam_mean = np.mean(results['sr_oam'])
    psnr_auth_mean = np.mean(results['psnr_auth'])
    if psnr_auth_mean < 25:
        print("  ⚠ 警告: 合法 PSNR_C < 25 dB, 模型训练质量不足")
    if sr_rpp_mean >= 0.3:
        print(f"  ⚠ 警告: RPP 攻击 SecurityRatio {sr_rpp_mean:.3f} >= 0.3 (未达安全目标)")
    else:
        print(f"  ✓ RPP 攻击 SecurityRatio {sr_rpp_mean:.3f} < 0.3 (通过)")
    if sr_oam_mean >= 0.3:
        print(f"  ⚠ 警告: OAM 攻击 SecurityRatio {sr_oam_mean:.3f} >= 0.3 (未达安全目标)")
    else:
        print(f"  ✓ OAM 攻击 SecurityRatio {sr_oam_mean:.3f} < 0.3 (通过)")
    print("=" * 70)


def plot_security(results, save_path="security_ratio_10ch.png"):
    """画 PSNR + SecurityRatio 分布图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (1) PSNR_C 分布
    ax = axes[0]
    bins = np.arange(0, 35, 2)
    ax.hist(results['psnr_auth'], bins=bins, alpha=0.6, label='合法解密', color='green')
    ax.hist(results['psnr_rpp'], bins=bins, alpha=0.6, label='RPP 攻击', color='orange')
    ax.hist(results['psnr_oam'], bins=bins, alpha=0.6, label='OAM 攻击', color='red')
    ax.axvline(25, color='gray', linestyle='--', label='25 dB 阈值')
    ax.set_xlabel("PSNR_C (dB)")
    ax.set_ylabel("频次")
    ax.set_title("10 通道 PSNR_C 分布")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (2) SecurityRatio 分布
    ax = axes[1]
    bins = np.arange(0, 1.5, 0.05)
    ax.hist(results['sr_rpp'], bins=bins, alpha=0.6, label='RPP 攻击', color='orange')
    ax.hist(results['sr_oam'], bins=bins, alpha=0.6, label='OAM 攻击', color='red')
    ax.axvline(0.3, color='gray', linestyle='--', label='0.3 阈值 (目标)')
    ax.set_xlabel("SecurityRatio")
    ax.set_ylabel("频次")
    ax.set_title("10 通道 SecurityRatio 分布")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("10 通道 OAM-MDNN 安全测试", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n安全测试图已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="oam_crypt_dnn_epoch_5.pth",
                        help="checkpoint 路径 (默认: oam_crypt_dnn_epoch_5.pth)")
    parser.add_argument("--n_test", type=int, default=40, help="测试样本数")
    parser.add_argument("--out", type=str, default="security_ratio_10ch.png", help="输出图路径")
    args = parser.parse_args()

    device = CONFIG["device"]

    # 1. 准备测试数据
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    full_test = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    subset = Subset(full_test, range(args.n_test))
    num_channels = len(CONFIG["l_auth"])
    img_size = CONFIG["size"] // 5
    from oam_crypt_d2nn import MNISTQuadDataset
    test_dataset = MNISTQuadDataset(subset, img_size=img_size, num_channels=num_channels)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 2. 加载模型
    print(f"[1/3] 加载 checkpoint: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        # 自动选最新的
        import glob
        ckpts = glob.glob("oam_crypt_dnn_epoch_*.pth")
        if not ckpts:
            print(f"  ✗ 找不到任何 checkpoint")
            return
        args.checkpoint = max(ckpts, key=os.path.getmtime)
        print(f"  自动选择最新: {args.checkpoint}")
    model, rpp_system = load_model(args.checkpoint, device)
    print(f"  ✓ 模型已加载")

    # 3. 攻击测试
    print(f"\n[2/3] 攻击测试 (n={args.n_test})...")
    results = run_attack_test(model, rpp_system, test_loader, device)

    # 4. 报告
    print_report(results)
    plot_security(results, save_path=args.out)

    # 5. 保存原始数据
    np.savez("security_ratio_10ch_data.npz", **results)
    print(f"\n原始数据已保存: security_ratio_10ch_data.npz")


if __name__ == "__main__":
    main()
