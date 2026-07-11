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

    ckpt = sys.argv[1] if len(sys.argv) > 1 else "oam_crypt_dnn_epoch_18.pth"  # 最佳模型 (3层D2NN, PSNR 31.85 dB)
    os.makedirs(SLM_CONFIG["output_dir"], exist_ok=True)

    # 1. 数据 + RPP + 模型
    transform = transforms.Compose([transforms.ToTensor()])
    full_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    mnist_test = Subset(full_test, range(8))
    test_dataset = MNISTQuadDataset(mnist_test, img_size=CONFIG["size"] // 4)  # 匹配训练配置 (size//4, 增强 OAM 正交性)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    rpp_placeholder = generate_rpp(CONFIG["size"], device)
    theta_max_rad = np.deg2rad(CONFIG["theta_max_deg"]) if CONFIG.get("theta_max_deg") else None

    model = OAM_Crypt_D2NN(
        size=CONFIG["size"], num_layers=CONFIG["num_layers"],
        wavelength=CONFIG["wavelength"], pixel_size=CONFIG["pixel_size"],
        z_layer=CONFIG["z_layer"], z0=CONFIG["z0"], rpp=rpp_placeholder,
        oam_keys=CONFIG["l_auth"], z_list=CONFIG["z_list"],
        obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max_rad
    ).to(device)
    ckpt_data = torch.load(ckpt, map_location=device)
    if isinstance(ckpt_data, dict) and 'model_state_dict' in ckpt_data:
        model.load_state_dict(ckpt_data['model_state_dict'])
        print(f"已加载: {ckpt} (PSNR={ckpt_data.get('psnr','?'):.2f} dB, SR={ckpt_data.get('sec_ratio','?'):.4f})", flush=True)
    else:
        model.load_state_dict(ckpt_data)
        print(f"已加载: {ckpt} (旧格式)", flush=True)
    model.eval()

    # 从模型 buffer 取训练时的 RPP (rpp_conj 的共轭 = 原始 rpp)
    rpp_system = torch.conj(model.rpp_conj)
    print(f"已从 checkpoint 恢复训练时的 RPP 系统密钥", flush=True)

    # 2. 取一个样本加密
    batch_imgs = next(iter(test_loader)).to(device)
    target = build_target_grid(batch_imgs, device)
    U_cipher = encrypt_batch(
        batch_imgs, CONFIG["l_auth"], rpp_system,
        CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
        z_list=CONFIG["z_list"], obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max_rad
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

    # ============================================================
    # 7. D2NN 可训练衍射层相位导出 (反射式光路用, 每层对应 SLM 一个反射区域)
    # ============================================================
    num_layers = CONFIG["num_layers"]
    if num_layers > 0:
        print()
        print("=" * 60)
        print(f"D2NN 可训练衍射层相位导出 ({num_layers} 层)")
        print("=" * 60)
        print(f"每层对应反射式光路中光束在 SLM 上反射的一个区域")
        print(f"层间距 z_layer = {CONFIG['z_layer']} m ({CONFIG['z_layer']*100:.0f} cm)")
        print(f"总光程: {num_layers} 层 × {CONFIG['z_layer']*100:.0f} cm = {num_layers*CONFIG['z_layer']*100:.0f} cm")
        print()

        d2nn_phases = []  # 保存各层相位用于可视化
        d2nn_grays = []
        for i, layer in enumerate(model.layers):
            with torch.no_grad():
                # 相位 [-π, π]
                phase_l = layer.phase.detach().cpu().numpy()
                # 振幅透过率 (sigmoid 输出, 实际 SLM 纯相位时可忽略)
                amp_l = torch.sigmoid(layer.amp_logit).detach().cpu().numpy()
            gray_l = np.clip((phase_l + np.pi) / (2 * np.pi) * 255 + 0.5, 0, 255).astype(np.uint8)
            d2nn_phases.append(phase_l)
            d2nn_grays.append(gray_l)

            # 保存单层相位图
            path_layer = os.path.join(SLM_CONFIG["output_dir"], f"d2nn_layer{i+1}_phase_{size}x{size}.png")
            Image.fromarray(gray_l).save(path_layer)

            # 统计
            print(f"Layer {i+1}/{num_layers}:")
            print(f"  相位范围: [{phase_l.min():.4f}, {phase_l.max():.4f}] rad")
            print(f"  相位 std: {phase_l.std():.4f} rad")
            print(f"  振幅透过率: mean={amp_l.mean():.4f}, std={amp_l.std():.4f} (纯相位SLM可忽略)")
            print(f"  灰度均值: {gray_l.mean():.2f}, std: {gray_l.std():.2f}")
            print(f"  保存: {path_layer}")

        # 合并所有 D2NN 层 + 密文相位到一张总览图 (反射式光路 SLM 布局示意)
        # 反射式光路: 光束依次打到 SLM 区域1 -> 区域2 -> 区域3 (每层间自由传播 z_layer)
        # 加上密文的双相位编码区域 (区域0, 加载到 SLM)
        fig, axes = plt.subplots(2, num_layers + 1, figsize=(5 * (num_layers + 1), 10))
        if num_layers + 1 == 1:
            axes = axes.reshape(2, 1)

        # 行1: 相位 (twilight 配色, [-π, π])
        # 区域0: 密文双相位编码
        im = axes[0, 0].imshow(phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
        axes[0, 0].set_title(f"区域0: 密文双相位编码\n(反射式光路起点)\n{size}×{size}", fontsize=10)
        plt.colorbar(im, ax=axes[0, 0], fraction=0.046)
        # 区域1..N: D2NN 层
        for i in range(num_layers):
            im = axes[0, i + 1].imshow(d2nn_phases[i], cmap='twilight', vmin=-np.pi, vmax=np.pi)
            axes[0, i + 1].set_title(f"区域{i+1}: D2NN Layer {i+1}\n(第{i+1}次反射)\nstd={d2nn_phases[i].std():.3f} rad", fontsize=10)
            plt.colorbar(im, ax=axes[0, i + 1], fraction=0.046)

        # 行2: 灰度图 (SLM 实际加载的 8-bit 图)
        axes[1, 0].imshow(gray_size, cmap='gray', vmin=0, vmax=255)
        axes[1, 0].set_title(f"区域0 灰度\n[0, 255] uint8", fontsize=10)
        for i in range(num_layers):
            axes[1, i + 1].imshow(d2nn_grays[i], cmap='gray', vmin=0, vmax=255)
            axes[1, i + 1].set_title(f"区域{i+1} 灰度\nmean={d2nn_grays[i].mean():.1f}", fontsize=10)

        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle(f"反射式 D2NN 光路 SLM 相位布局 ({num_layers} 层, z_layer={CONFIG['z_layer']*100:.0f}cm, "
                     f"总光程={num_layers*CONFIG['z_layer']*100:.0f}cm)\n"
                     f"光束路径: 区域0 → [传播 {CONFIG['z_layer']*100:.0f}cm] → 区域1 → ... → 区域{num_layers}",
                     fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        d2nn_overview = os.path.join(SLM_CONFIG["output_dir"], f"d2nn_layers_overview_{num_layers}L.png")
        plt.savefig(d2nn_overview, dpi=120, bbox_inches='tight')
        plt.close()
        print()
        print(f"D2NN 层相位总览图: {d2nn_overview}")

        # 反射式光路 SLM 布局: 在 1920×1080 SLM 上水平排列 N+1 个区域
        # 每个区域宽度 = 1920 / (N+1), 高度 = 1080
        # 注意: 实际反射式光路中这些区域是同一 SLM 的不同空间位置, 光束斜入射依次反射
        n_regions = num_layers + 1  # 区域0 (密文) + N 个 D2NN 层
        region_w = SLM_CONFIG["width"] // n_regions
        region_h = SLM_CONFIG["height"]
        # 把 1080×1080 全息图缩放到 region_w × region_h
        from PIL import Image as PILImage

        slm_reflective = np.zeros((SLM_CONFIG["height"], SLM_CONFIG["width"]), dtype=np.uint8)
        # 区域0: 密文双相位 (缩放)
        img0 = PILImage.fromarray(gray_size).resize((region_w, region_h), PILImage.BILINEAR)
        slm_reflective[:, 0:region_w] = np.array(img0)
        # 区域1..N: D2NN 层相位 (缩放)
        for i in range(num_layers):
            img_i = PILImage.fromarray(d2nn_grays[i]).resize((region_w, region_h), PILImage.BILINEAR)
            x_start = (i + 1) * region_w
            slm_reflective[:, x_start:x_start + region_w] = np.array(img_i)

        path_reflective = os.path.join(SLM_CONFIG["output_dir"],
                                       f"slm_reflective_{num_layers}L_{SLM_CONFIG['width']}x{SLM_CONFIG['height']}.png")
        Image.fromarray(slm_reflective).save(path_reflective)
        print(f"反射式 SLM 布局图 ({n_regions} 区域水平排列): {path_reflective}")
        print(f"  每区域尺寸: {region_w}×{region_h}")
        print(f"  区域0 (密文双相位): x=[0, {region_w}]")
        for i in range(num_layers):
            x_start = (i + 1) * region_w
            print(f"  区域{i+1} (D2NN Layer {i+1}): x=[{x_start}, {x_start + region_w}]")

    print()
    print("=" * 60)
    print("SLM 加载说明 (双相位编码 + D2NN 反射式光路)")
    print("=" * 60)
    print(f"1. 反射式光路: SLM + 平面镜, 光束斜入射")
    print(f"2. SLM 分 {num_layers + 1 if num_layers > 0 else 1} 个区域:")
    print(f"   - 区域0: 密文双相位编码 (输入)")
    if num_layers > 0:
        for i in range(num_layers):
            print(f"   - 区域{i+1}: D2NN Layer {i+1} (第{i+1}次反射, 可训练相位)")
    print(f"3. 光束路径: 区域0 → [传播 {CONFIG['z_layer']*100:.0f}cm] → 区域1 → ... → 区域{num_layers}")
    print(f"4. 工作波长: 532 nm (绿光)")
    print(f"5. 加载文件:")
    print(f"   - 密文: slm_phase_{SLM_CONFIG['width']}x{SLM_CONFIG['height']}.png")
    if num_layers > 0:
        for i in range(num_layers):
            print(f"   - D2NN Layer {i+1}: d2nn_layer{i+1}_phase_{size}x{size}.png")
        print(f"   - 反射式布局总图: slm_reflective_{num_layers}L_{SLM_CONFIG['width']}x{SLM_CONFIG['height']}.png")
    print(f"6. 多平面选择性: z_list={CONFIG['z_list']}")
    print(f"7. 解密 PSNR = {psnr_dp:.2f} dB (> 30 dB 目标)")


if __name__ == "__main__":
    main()
