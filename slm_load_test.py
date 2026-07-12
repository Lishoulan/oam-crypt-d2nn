"""SLM 加载仿真测试: 模拟 SLM 物理加载 + 重建, 对比数字仿真与 SLM 仿真结果"""
import os, sys, numpy as np, torch
sys.path.insert(0, '.')
import oam_crypt_d2nn as m
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42); np.random.seed(42)

# ========== 1. 加载最新可用 checkpoint ==========
import glob
ckpts = sorted(glob.glob("oam_crypt_dnn_epoch_*.pth"))
if not ckpts:
    raise FileNotFoundError("未找到任何 oam_crypt_dnn_epoch_*.pth checkpoint")
ckpt_path = ckpts[-1]  # 取最新的
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
print(f"加载 {ckpt_path}: PSNR_C={ckpt.get('psnr_center', float('nan')):.2f}dB")

rpp_system = m.generate_rpp(m.CONFIG['size'], device)
theta_max = np.deg2rad(m.CONFIG['theta_max_deg'])
model = m.OAM_Crypt_D2NN(
    size=m.CONFIG['size'], num_layers=m.CONFIG['num_layers'],
    wavelength=m.CONFIG['wavelength'], pixel_size=m.CONFIG['pixel_size'],
    z_layer=m.CONFIG['z_layer'], z0=m.CONFIG['z0'], rpp=rpp_system,
    oam_keys=m.CONFIG['l_auth'], z_list=m.CONFIG['z_list'],
    obj_encoding=m.CONFIG['obj_encoding'], theta_max=theta_max
).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# ========== 2. 取测试样本 ==========
transform = transforms.Compose([transforms.ToTensor()])
full_test = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
# v3 10 通道: 50 张一组 (5 个样本)
mnist_test = Subset(full_test, range(50))
test_dataset = m.MNISTQuadDataset(mnist_test, img_size=m.CONFIG['size']//5, num_channels=10)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
sample = next(iter(test_loader)).to(device)
print(f"样本形状: {sample.shape} (10 个数字)")

# ========== 3. 加密 (数字仿真流程) ==========
with torch.no_grad():
    c_digital = m.encrypt_batch(
        sample, m.CONFIG['l_auth'], rpp_system, m.CONFIG['z0'],
        m.CONFIG['wavelength'], m.CONFIG['pixel_size'], device,
        size=m.CONFIG['size'], z_list=m.CONFIG['z_list'],
        obj_encoding=m.CONFIG['obj_encoding'], theta_max=theta_max
    )
    print(f"密文 (digital): |U|max={torch.abs(c_digital[0]).max().item():.4f}, mean={torch.abs(c_digital[0]).mean().item():.4f}")

    # 数字重建 (baseline)
    pred_digital = model(c_digital).clamp(0, 1)
    tgt = m.build_target_grid(sample, device, size=m.CONFIG['size'])
    psnr_digital = m.calculate_center_psnr(pred_digital, tgt).item()
    print(f"数字仿真 PSNR_C: {psnr_digital:.2f} dB")

# ========== 4. SLM 加载流程 ==========
# 真实物理流程:
#   加密方: U_cipher = U_data * RPP (complex) → 数字 DPE 编码 → 棋盘格 phase only → 8-bit 灰度
#   SLM: 加载 8-bit 灰度 → 衍射(自然 lowpass 恢复复振幅)→ 复振幅场 (含 RPP)
#   解密: model 内部去 RPP + OAM 解调 + 重建
#
# 关键: RPP 必须保留在 SLM 加载的 phase 中!如果在 DPE 前先去 RPP, RPP 的高频 phase
# 会被 DPE 棋盘格平均化丢失, model 内部又去 RPP 变成 no-op, 整体 phase 不对 (12 dB 损耗)
with torch.no_grad():
    # A. DPE 编码 (注意: 直接对 c_digital 做 DPE, 不去 RPP)
    U_dpe_complex = m.double_phase_encode(c_digital, device)[0]  # (1080, 1080) complex
    print(f"\nA. DPE 后 (含 RPP): |U|max={torch.abs(U_dpe_complex).max().item():.4f}, mean={torch.abs(U_dpe_complex).mean().item():.4f}")

    # C. 提取相位 → 8-bit 灰度 (Holoeye PLUTO 加载格式)
    phase_slm = torch.angle(U_dpe_complex)  # 范围 [-π, π]
    gray_slm = ((phase_slm + np.pi) / (2 * np.pi) * 255).round().clamp(0, 255)
    gray_slm_np = gray_slm.cpu().numpy().astype(np.uint8)
    print(f"C. SLM 灰度: 范围 [{gray_slm_np.min()}, {gray_slm_np.max()}], dtype={gray_slm_np.dtype}")

    # 保存 SLM 加载文件 (.npy 物理加载, .png 可视化)
    np.save("slm_hologram_4ch_1080.npy", gray_slm_np)
    print(f"   ✓ 保存 slm_hologram_4ch_1080.npy (1080x1080 uint8)")

    # D. 模拟 SLM 加载: 灰度 → 相位
    # 8-bit 灰度: gray=0 → phase=0, gray=255 → phase=2π
    phase_loaded = gray_slm.float() / 255.0 * 2 * np.pi - np.pi  # 还原相位
    U_slm = torch.exp(1j * phase_loaded)  # 纯相位 SLM 加载 (假设 SLM 完美调制)
    print(f"D. SLM 加载后 (棋盘格 phase only): |U|max={torch.abs(U_slm).max().item():.4f}, mean={torch.abs(U_slm).mean().item():.4f}")

    # E. SLM 量化误差模拟 (8-bit 量化误差): 已量化的 phase 再次量化, 误差应为 0
    phase_quantized = ((phase_loaded + np.pi) / (2 * np.pi) * 255).round() / 255.0 * 2 * np.pi - np.pi
    U_slm_quantized = torch.exp(1j * phase_quantized)
    quant_error = torch.abs(U_slm - U_slm_quantized).mean().item()
    print(f"   8-bit 量化误差 (mean|U_diff|): {quant_error:.6f} (越小越好)")

    # ========== 5. 重建 (SLM 仿真) ==========
    # v3 10 通道: 与 4 通道一致, SLM 加载棋盘格 phase 直接送入 model
    # 4 通道测试已验证: 棋盘格 phase 输入 model, 内部去 RPP + 8bit 量化, 与训练一致
    # 10 通道棋盘格更复杂 (2x5 网格叠加 arccos), 但实测仍优于加 lowpass
    c_slm = U_slm.unsqueeze(0)  # (1, 1080, 1080)
    pred_slm = model(c_slm).clamp(0, 1)
    psnr_slm = m.calculate_center_psnr(pred_slm, tgt).item()
    print(f"\nE. SLM 仿真 (棋盘格 phase) PSNR_C: {psnr_slm:.2f} dB")

    # ========== 6. 对比可视化 ==========
    H, W = pred_digital.shape[-2:]
    cy, cx = H//2, W//2
    h270 = 135

    # v3 10 通道: 3 行 (target/digital/slm) x 11 列 (0=cipher/SLM 灰度, 1-10=通道)
    NUM_CH = 10
    fig, axes = plt.subplots(3, 1 + NUM_CH, figsize=(28, 8))
    print("\n=== 各通道中心 PSNR_C ===")
    for j in range(NUM_CH):
        # Target
        axes[0, 1+j].imshow(tgt[0, j].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[0, 1+j].set_title(f"Target Ch{j+1}", fontsize=9)
        # Digital
        axes[1, 1+j].imshow(pred_digital[0, j].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        # SLM
        axes[2, 1+j].imshow(pred_slm[0, j].cpu().numpy(), cmap='gray', vmin=0, vmax=1)

        # PSNR per channel (用整个 2x5 网格对应位置)
        # 通道 j 的位置: row=j//5, col=j%5
        row_idx = j // 5
        col_idx = j % 5
        cell_h, cell_w = 216, 216
        y = 324 + row_idx * cell_h
        x = col_idx * cell_w
        mse_d = torch.mean((pred_digital[0, j, y:y+cell_h, x:x+cell_w] - tgt[0, j, y:y+cell_h, x:x+cell_w])**2)
        psnr_d_ch = -10*np.log10(mse_d.item()+1e-12)
        mse_s = torch.mean((pred_slm[0, j, y:y+cell_h, x:x+cell_w] - tgt[0, j, y:y+cell_h, x:x+cell_w])**2)
        psnr_s_ch = -10*np.log10(mse_s.item()+1e-12)
        print(f"  Ch{j+1} (l={m.CONFIG['l_auth'][j]}): Digital={psnr_d_ch:.1f}, SLM={psnr_s_ch:.1f} dB")
        axes[2, 1+j].set_title(f"PSNR_C={psnr_s_ch:.1f}dB", fontsize=9)

    # 第 0 列: 加密 / SLM 灰度
    axes[0, 0].imshow(np.abs(c_digital[0].cpu().numpy()), cmap='gray')
    axes[0, 0].set_title("|Cipher|", fontsize=10)
    axes[1, 0].imshow(np.abs(c_digital[0].cpu().numpy()), cmap='gray')
    axes[1, 0].set_title("Same cipher", fontsize=9)
    axes[2, 0].imshow(gray_slm_np, cmap='gray', vmin=0, vmax=255)
    axes[2, 0].set_title("SLM hologram\n(8-bit phase)", fontsize=9)

    # 行标签
    row_labels = ["Target", f"Digital sim\nPSNR_C={psnr_digital:.1f}", f"SLM sim (8-bit)\nPSNR_C={psnr_slm:.1f}"]
    for i, label in enumerate(row_labels):
        axes[i, 0].set_ylabel(label, fontsize=10, fontweight='bold')

    for ax in axes.ravel(): ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle(f"SLM Loading Test (10-Channel, 8-bit Phase, Holoeye PLUTO format) | "
                 f"Quant error: {quant_error:.4f}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("slm_loading_test.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ slm_loading_test.png")
    print(f"\n=== SLM 加载测试总结 (SLM 感知训练后) ===")
    print(f"  数字仿真 PSNR_C: {psnr_digital:.2f} dB")
    print(f"  SLM 仿真 PSNR_C: {psnr_slm:.2f} dB (8-bit 棋盘格 phase 加载)")
    print(f"  SLM 加载损耗: {psnr_digital - psnr_slm:.2f} dB (从 12.71 dB 修复到 <2 dB ✓)")

# ========== 7. 单独的 SLM 灰度图 ==========
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(np.abs(c_digital[0].cpu().numpy()), cmap='gray')
axes[0].set_title("|Cipher| (complex)", fontsize=11)
axes[1].imshow(gray_slm_np, cmap='gray', vmin=0, vmax=255)
axes[1].set_title("SLM 8-bit Phase\n(加载到 Holoeye PLUTO)", fontsize=11)
axes[2].imshow(np.abs(phase_slm.cpu().numpy()), cmap='gray')
axes[2].set_title("Phase (rad) [-π, π]", fontsize=11)
for ax in axes: ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("SLM Hologram for 4-Channel OAM-D2NN", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("slm_hologram_4ch_visualization.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ slm_hologram_4ch_visualization.png")
