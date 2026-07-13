# -*- coding: utf-8 -*-
"""
v8 结果可视化:
1) Stage 1 2 通道解密 vs 攻击对比 (一张大图: 明文 | 合法 | RPP 攻击 | OAM 攻击)
2) Stage 1 SLM 加载前后对比
3) v8 4 stage PSNR_C 训练曲线
"""
import sys, os
import numpy as np
import torch
sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from font_config import setup_cjk
setup_cjk()

from oam_crypt_d2nn import (
    CONFIG, OAM_Crypt_D2NN, generate_rpp, encrypt_batch, build_target_grid,
    calculate_center_psnr, MNISTQuadDataset
)
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

device = torch.device(CONFIG['device'])
torch.manual_seed(42)
np.random.seed(42)

print("="*80)
print("[v8 可视化] PolarHNN 结果图生成")
print("="*80)

# ----------------------------------------------------------------------
# 1) 加载 v8 stage 1 (2 通道) 模型
# ----------------------------------------------------------------------
l_auth = [-25, 25]
z_list = [0.10, 0.55]
l_wrong = [-3, 3]
rpp = generate_rpp(CONFIG['size'], device, generator=torch.Generator(device).manual_seed(42))
rpp_wrong = generate_rpp(CONFIG['size'], device, generator=torch.Generator(device).manual_seed(123))

ckpt = torch.load("oam_crypt_v8_stage1_best.pth", map_location=device, weights_only=False)
print(f"[加载] stage1_best: PSNR_C={ckpt['psnr_center']:.2f} dB")

model = OAM_Crypt_D2NN(
    size=CONFIG['size'], num_layers=CONFIG['num_layers'],
    wavelength=CONFIG['wavelength'], pixel_size=CONFIG['pixel_size'],
    z_layer=CONFIG['z_layer'], z0=CONFIG['z0'], rpp=rpp,
    oam_keys=l_auth, z_list=z_list,
    obj_encoding=CONFIG['obj_encoding'],
    theta_max=np.deg2rad(CONFIG['theta_max_deg']) if CONFIG.get('theta_max_deg') else None,
    slm_aware=True,  # 匹配训练时
    use_channel_attn=True, mid_ch=CONFIG['mid_ch'],
    iterative_refine=False,
    oam_freq_filter=True,
    use_polar_conv=True,
    polar_n_r=32, polar_n_theta=96,
).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# ----------------------------------------------------------------------
# 2) 准备数据 (MNIST test 集前 20 个, 每 2 张 1 样本)
# ----------------------------------------------------------------------
transform = transforms.Compose([transforms.ToTensor()])
full_test = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
mnist_sub = Subset(full_test, range(0, 20))  # 20 个 MNIST -> 10 个 MNISTQuadDataset 样本
ds = MNISTQuadDataset(mnist_sub, img_size=CONFIG['size']//5, num_channels=2)
loader = DataLoader(ds, batch_size=1, shuffle=False)

# 取前 4 个样本做对比
samples = []
for i, b in enumerate(loader):
    if i >= 4: break
    samples.append(b.to(device))

# ----------------------------------------------------------------------
# 3) 计算 4 种情况: 明文 / 合法 / RPP 攻击 / OAM 攻击
# ----------------------------------------------------------------------
theta_max = np.deg2rad(CONFIG['theta_max_deg'])
def encrypt(imgs, lk, rppk):
    return encrypt_batch(
        imgs, lk, rppk,
        CONFIG['z0'], CONFIG['wavelength'], CONFIG['pixel_size'], device,
        size=CONFIG['size'], z_list=z_list,
        obj_encoding=CONFIG['obj_encoding'], theta_max=theta_max,
        layout='oam_overlap'
    )

results = []  # 每个样本: {plain, auth, rpp_atk, oam_atk, psnrs}
for i, imgs in enumerate(samples):
    target = build_target_grid(imgs, device, size=CONFIG['size'], layout='oam_overlap')
    # 合法
    cipher_auth = encrypt(imgs, l_auth, rpp)
    with torch.no_grad():
        pred_auth = model(cipher_auth).clamp(0, 1)
    psnr_auth = calculate_center_psnr(pred_auth, target).item()
    # RPP 攻击
    cipher_rpp = encrypt(imgs, l_auth, rpp_wrong)
    with torch.no_grad():
        pred_rpp = model(cipher_rpp).clamp(0, 1)
    psnr_rpp = calculate_center_psnr(pred_rpp, target).item()
    # OAM 攻击
    cipher_oam = encrypt(imgs, l_wrong, rpp)
    with torch.no_grad():
        pred_oam = model(cipher_oam).clamp(0, 1)
    psnr_oam = calculate_center_psnr(pred_oam, target).item()
    results.append({
        'plain': target[0],  # (2, 1080, 1080)
        'auth': pred_auth[0], 'psnr_auth': psnr_auth,
        'rpp': pred_rpp[0], 'psnr_rpp': psnr_rpp,
        'oam': pred_oam[0], 'psnr_oam': psnr_oam,
    })
    print(f"  样本 {i+1}: 合法 {psnr_auth:.2f} | RPP 攻击 {psnr_rpp:.2f} | OAM 攻击 {psnr_oam:.2f} dB")

# 中心区域提取
def center_crop(img, half=108):
    """img: (2, 1080, 1080) -> 拼接成 (216, 432) (2 通道横向)"""
    c0 = img[0, 540-half:540+half, 540-half:540+half].cpu().numpy()
    c1 = img[1, 540-half:540+half, 540-half:540+half].cpu().numpy()
    return np.concatenate([c0, c1], axis=1)  # (216, 432)

# ----------------------------------------------------------------------
# 4) 生成大对比图: 4 行(样本) x 4 列(明文/合法/RPP 攻击/OAM 攻击)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(len(results), 4, figsize=(16, 4 * len(results)))
col_titles = ['明文 (Target)', '合法解密', 'RPP 攻击', 'OAM 攻击 (-3, +3)']
for j, t in enumerate(col_titles):
    axes[0, j].set_title(t, fontsize=14, fontweight='bold')

for i, r in enumerate(results):
    plain = center_crop(r['plain'])
    auth = center_crop(r['auth'])
    rpp_img = center_crop(r['rpp'])
    oam_img = center_crop(r['oam'])
    axes[i, 0].imshow(plain, cmap='gray', vmin=0, vmax=1)
    axes[i, 0].set_ylabel(f'样本 {i+1}', fontsize=12)
    axes[i, 1].imshow(auth, cmap='gray', vmin=0, vmax=1)
    axes[i, 1].set_xlabel(f'PSNR_C = {r["psnr_auth"]:.2f} dB', fontsize=11, color='green')
    axes[i, 2].imshow(rpp_img, cmap='gray', vmin=0, vmax=1)
    axes[i, 2].set_xlabel(f'PSNR_C = {r["psnr_rpp"]:.2f} dB', fontsize=11, color='red')
    axes[i, 3].imshow(oam_img, cmap='gray', vmin=0, vmax=1)
    axes[i, 3].set_xlabel(f'PSNR_C = {r["psnr_oam"]:.2f} dB', fontsize=11, color='red')
    for j in range(4):
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])

plt.suptitle('v8 PolarHNN Stage 1 (2 通道 oam_overlap) — 解密 vs 攻击对比',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("v8_stage1_attack_comparison.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"[图1] 已保存: v8_stage1_attack_comparison.png")

# ----------------------------------------------------------------------
# 5) 生成 SLM 加载对比图
# ----------------------------------------------------------------------
model_digital = OAM_Crypt_D2NN(
    size=CONFIG['size'], num_layers=CONFIG['num_layers'],
    wavelength=CONFIG['wavelength'], pixel_size=CONFIG['pixel_size'],
    z_layer=CONFIG['z_layer'], z0=CONFIG['z0'], rpp=rpp,
    oam_keys=l_auth, z_list=z_list,
    obj_encoding=CONFIG['obj_encoding'],
    theta_max=np.deg2rad(CONFIG['theta_max_deg']) if CONFIG.get('theta_max_deg') else None,
    slm_aware=False,  # 数字加载
    use_channel_attn=True, mid_ch=CONFIG['mid_ch'],
    iterative_refine=False, oam_freq_filter=True,
    use_polar_conv=True, polar_n_r=32, polar_n_theta=96,
).to(device)
ckpt_d = torch.load("oam_crypt_v8_stage1_best.pth", map_location=device, weights_only=False)
model_digital.load_state_dict(ckpt_d['model_state_dict'])
model_digital.eval()

fig, axes = plt.subplots(len(results), 3, figsize=(12, 4 * len(results)))
col_titles = ['明文', '数字加载 (no SLM)', 'SLM 8-bit 加载']
for j, t in enumerate(col_titles):
    axes[0, j].set_title(t, fontsize=14, fontweight='bold')

for i, imgs in enumerate(samples):
    target = build_target_grid(imgs, device, size=CONFIG['size'], layout='oam_overlap')
    cipher = encrypt(imgs, l_auth, rpp)
    with torch.no_grad():
        pred_d = model_digital(cipher).clamp(0, 1)
        pred_s = model(cipher).clamp(0, 1)
    psnr_d = calculate_center_psnr(pred_d, target).item()
    psnr_s = calculate_center_psnr(pred_s, target).item()
    plain = center_crop(target[0])  # 去 batch 维
    d_img = center_crop(pred_d[0])
    s_img = center_crop(pred_s[0])
    axes[i, 0].imshow(plain, cmap='gray', vmin=0, vmax=1)
    axes[i, 0].set_ylabel(f'样本 {i+1}', fontsize=12)
    axes[i, 1].imshow(d_img, cmap='gray', vmin=0, vmax=1)
    axes[i, 1].set_xlabel(f'PSNR_C = {psnr_d:.2f} dB', fontsize=11)
    axes[i, 2].imshow(s_img, cmap='gray', vmin=0, vmax=1)
    axes[i, 2].set_xlabel(f'PSNR_C = {psnr_s:.2f} dB  (损耗 {psnr_d - psnr_s:+.2f} dB)', fontsize=11)
    for j in range(3):
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])

plt.suptitle('v8 Stage 1 — SLM 8-bit 加载鲁棒性 (8GB GPU, slm_aware 训练)',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("v8_stage1_slm_loading.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"[图2] 已保存: v8_stage1_slm_loading.png")

# ----------------------------------------------------------------------
# 6) 生成 4 stage PSNR_C 训练曲线
# ----------------------------------------------------------------------
# 从 run_v8_log.txt 解析
import re
psnr_data = {
    'Stage 1 (2 通道)':  [],
    'Stage 2 (5 通道)':  [],
    'Stage 3 (8 通道)':  [],
    'Stage 4 (10 通道)': [],
}
log_path = "run_v8_log.txt"
if not os.path.exists(log_path):
    log_path = "run_v8_quick_log.txt"
if os.path.exists(log_path):
    with open(log_path, 'r', encoding='utf-8') as f:
        log = f.read()
    # 解析 [S{n} E{m}/{total}] ... PSNR_C={x} dB
    pattern = re.compile(r"\[S(\d+) E\d+/\d+\] .*PSNR_C=([\d.]+) dB")
    matches = pattern.findall(log)
    current_stage = 0
    for s, p in matches:
        stage_idx = int(s) - 1
        if 0 <= stage_idx <= 3:
            psnr_data[f'Stage {stage_idx+1} ({["2","5","8","10"][stage_idx]} 通道)'].append(float(p))
    # Stage 4 用完 14 epoch, 前面被截断的也要全捕
    print(f"[日志解析] Stage 1: {len(psnr_data['Stage 1 (2 通道)'])} epochs")
    print(f"[日志解析] Stage 2: {len(psnr_data['Stage 2 (5 通道)'])} epochs")
    print(f"[日志解析] Stage 3: {len(psnr_data['Stage 3 (8 通道)'])} epochs")
    print(f"[日志解析] Stage 4: {len(psnr_data['Stage 4 (10 通道)'])} epochs")

fig, ax = plt.subplots(figsize=(12, 7))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
best_vals = []
for idx, (label, vals) in enumerate(psnr_data.items()):
    if len(vals) == 0:
        continue
    epochs = list(range(1, len(vals) + 1))
    ax.plot(epochs, vals, 'o-', linewidth=2, markersize=8, color=colors[idx], label=label)
    # 标 best
    best_idx = np.argmax(vals)
    best_val = vals[best_idx]
    best_vals.append((label, best_val, best_idx + 1))
    ax.annotate(f'Best {best_val:.2f}', xy=(best_idx + 1, best_val),
                xytext=(best_idx + 1, best_val + 0.5), fontsize=10,
                ha='center', color=colors[idx], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=colors[idx], alpha=0.5))
ax.axhline(y=20.0, color='red', linestyle='--', alpha=0.5, label='20 dB 工程目标')
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('PSNR_C (dB)', fontsize=14)
ax.set_title('v8 PolarHNN Curriculum 4 Stage 训练曲线 (8GB GPU, 79.7 min)', fontsize=15, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(list(range(1, 15)))
plt.tight_layout()
plt.savefig("v8_training_curves.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"[图3] 已保存: v8_training_curves.png")

# ----------------------------------------------------------------------
# 7) 生成 v8 vs v7 性能对比柱状图
# ----------------------------------------------------------------------
versions = ['v5', 'v6', 'v7', 'v8']
v2  = [14.17, 14.65, 22.89, 23.44]
v5  = [None, None, 17.80, 16.94]
v8  = [None, None, 13.74, 13.22]
v10 = [11.02, None, 13.88, 13.73]

fig, ax = plt.subplots(figsize=(10, 7))
x = np.arange(len(versions))
width = 0.2
data_groups = [
    ('2 通道',  [v if v is not None else 0 for v in v2],  '#1f77b4'),
    ('5 通道',  [v if v is not None else 0 for v in v5],  '#ff7f0e'),
    ('8 通道',  [v if v is not None else 0 for v in v8],  '#2ca02c'),
    ('10 通道', [v if v is not None else 0 for v in v10], '#d62728'),
]
for i, (label, vals, color) in enumerate(data_groups):
    bars = ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=color, alpha=0.85)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3, f'{val:.1f}',
                    ha='center', fontsize=9, fontweight='bold')

ax.axhline(y=20.0, color='red', linestyle='--', alpha=0.5, label='20 dB 工程目标')
ax.set_xticks(x)
ax.set_xticklabels(versions, fontsize=13)
ax.set_ylabel('PSNR_C (dB)', fontsize=13)
ax.set_title('v5 → v8 4 通道数性能对比 (oam_overlap 模式)', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("v8_vs_v7_comparison.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"[图4] 已保存: v8_vs_v7_comparison.png")

print("\n[v8 可视化] 4 张图已生成:")
print("  1. v8_stage1_attack_comparison.png  (4 样本 x 4 列: 明文|合法|RPP 攻击|OAM 攻击)")
print("  2. v8_stage1_slm_loading.png         (4 样本 x 3 列: 数字|SLM 8-bit|损耗)")
print("  3. v8_training_curves.png            (4 stage 训练曲线)")
print("  4. v8_vs_v7_comparison.png           (v5-v8 4 通道数性能对比)")
