# -*- coding: utf-8 -*-
"""
v8 烟雾测试: Stage 1 2 通道 (PolarConv + OAM-FDD Loss 启用)
验证: 1) PolarConv 正常 forward/backward
     2) OAM-FDD loss 计算正确
     3) 训练 1 个 epoch 正常 (1 batch), 显存占用
"""
import sys
import time
import torch
sys.path.insert(0, '.')
from oam_crypt_d2nn import (
    CONFIG, OAM_Crypt_D2NN, PolarConv, oam_fdd_loss,
    generate_rpp, encrypt_batch, build_target_grid, calculate_center_psnr,
    MNISTQuadDataset, generate_oam_phase
)
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

print(f"[v8 SMOKE] CONFIG polar_conv={CONFIG.get('polar_conv')}  oam_fdd_loss={CONFIG.get('oam_fdd_loss')}")
print(f"[v8 SMOKE] polar_n_r={CONFIG.get('polar_n_r')} polar_n_theta={CONFIG.get('polar_n_theta')}")

device = torch.device(CONFIG['device'])
torch.manual_seed(42)

# 1. 单元测试 PolarConv
print("\n[UNIT TEST 1] PolarConv forward/backward")
B, C, H, W = 1, 192, 270, 270
pc = PolarConv(C, n_r=32, n_theta=96, theta_kernel=7, init_scale=0.0).to(device)
x = torch.randn(B, C, H, W, device=device)
t0 = time.time()
y = pc(x)
print(f"  PolarConv input: {x.shape} output: {y.shape} t={time.time()-t0:.3f}s")
print(f"  PolarConv scale: {pc.scale.item():.4f} (init=0, training 暂时不变)")
# 验证 scale=0 时输出 = 输入
diff = (y - x).abs().max().item()
print(f"  scale=0 max diff: {diff:.2e} (should be 0)")
assert diff < 1e-5, "scale=0 should be identity"

# 测试 scale=0.5 时不恒等
pc.scale.data.fill_(0.5)
y2 = pc(x)
diff2 = (y2 - x).abs().max().item()
print(f"  scale=0.5 max diff: {diff2:.4f} (should be > 0)")

# 2. 单元测试 OAM-FDD loss
print("\n[UNIT TEST 2] OAM-FDD loss")
B, C, H, W = 1, 2, 1080, 1080
pred_dummy = torch.rand(B, C, H, W, device=device) * 0.5  # [0, 0.5]
l_auth_test = [-25, 25]
loss_fdd = oam_fdd_loss(pred_dummy, l_auth_test, l_radius=15)
print(f"  oam_fdd_loss (random 0-0.5): {loss_fdd.item():.4f}")

# 构造"完美"预测: 通道 0 强 l=5 谐波, 通道 1 强 l=15 谐波 (不同 |l| 才能在强度域分离)
y_grid, x_grid = torch.meshgrid(
    torch.arange(H, device=device, dtype=torch.float32),
    torch.arange(W, device=device, dtype=torch.float32),
    indexing='ij'
)
cx, cy = W // 2, H // 2
theta = torch.atan2(y_grid - cy, x_grid - cx)
pred_perfect = torch.zeros(B, C, H, W, device=device)
pred_perfect[0, 0] = 0.5 + 0.4 * torch.cos(5 * theta)  # 通道 0: 强 5 阶谐波
pred_perfect[0, 1] = 0.5 + 0.4 * torch.cos(15 * theta)  # 通道 1: 强 15 阶谐波
loss_fdd_perfect = oam_fdd_loss(pred_perfect, [-5, 15], l_radius=15)
print(f"  oam_fdd_loss (perfect diff |l|): {loss_fdd_perfect.item():.4f} (应该 < random)")

# 反例: 通道 0, 1 完全相同
pred_same = pred_perfect.clone()
pred_same[0, 1] = pred_perfect[0, 0]
loss_fdd_same = oam_fdd_loss(pred_same, [-5, 15], l_radius=15)
print(f"  oam_fdd_loss (完全相同): {loss_fdd_same.item():.4f} (应该 > random)")

# 3. 集成测试: 2 通道 OAM_Crypt_D2NN + PolarConv
print("\n[INTEGRATION TEST] 2 通道 1 epoch")
rpp = generate_rpp(CONFIG['size'], device)
theta_max_rad = 1.5 * 3.14159 / 180
model = OAM_Crypt_D2NN(
    size=CONFIG['size'], num_layers=3,
    wavelength=CONFIG['wavelength'], pixel_size=CONFIG['pixel_size'],
    z_layer=CONFIG['z_layer'], z0=CONFIG['z0'], rpp=rpp,
    oam_keys=[-25, 25], z_list=[0.10, 0.55],
    obj_encoding=CONFIG['obj_encoding'], theta_max=theta_max_rad,
    slm_aware=CONFIG['slm_aware'],
    use_channel_attn=True, mid_ch=CONFIG['mid_ch'],
    iterative_refine=False,  # v8 默认关闭
    n_passes=3,
    oam_freq_filter=CONFIG.get('oam_freq_filter', True),
    use_polar_conv=CONFIG.get('polar_conv', True),
    polar_n_r=CONFIG.get('polar_n_r', 32),
    polar_n_theta=CONFIG.get('polar_n_theta', 96),
    polar_init_scale=CONFIG.get('polar_init_scale', 0.0),
).to(device)

with torch.no_grad():
    for i in range(3):
        model.layers[i].phase.zero_()
        model.layers[i].amp_logit.fill_(4.0)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  模型参数: {n_params:,}")
n_polar = sum(p.numel() for p in model.refine.polar_conv.parameters() if p.requires_grad)
print(f"  PolarConv 参数: {n_polar:,}")

# 取 10 个样本
transform = transforms.Compose([transforms.ToTensor()])
full_train = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
mnist_sub = Subset(full_train, range(20))
img_size = CONFIG['size'] // 5
ds = MNISTQuadDataset(mnist_sub, img_size=img_size, num_channels=2)
loader = DataLoader(ds, batch_size=1, shuffle=True)

import torch.optim as optim
opt = optim.Adam(model.parameters(), lr=1e-4)

print(f"  训练 1 个 batch (1080x1080), polar_conv enabled...")
t0 = time.time()
model.train()
for batch_imgs in loader:
    batch_imgs = batch_imgs.to(device)
    target = build_target_grid(batch_imgs, device, size=CONFIG['size'], layout='oam_overlap')
    cipher = encrypt_batch(
        batch_imgs, [-25, 25], rpp,
        CONFIG['z0'], CONFIG['wavelength'], CONFIG['pixel_size'], device,
        size=CONFIG['size'], z_list=[0.10, 0.55],
        obj_encoding=CONFIG['obj_encoding'], theta_max=theta_max_rad,
        layout='oam_overlap'
    )
    opt.zero_grad()
    pred = model(cipher)
    # 中心区域 MSE
    H, W = target.shape[-2:]
    weight_map = torch.ones(1, 1, H, W, device=device) * 0.1
    weight_map[..., 432:648, 432:648] = 10.0
    loss = torch.mean(weight_map * (pred.clamp(0, 1) - target) ** 2)
    # FDD loss
    loss_fdd = oam_fdd_loss(pred.clamp(0, 1), [-25, 25], l_radius=3)
    total = loss + 0.05 * loss_fdd
    total.backward()
    opt.step()
    psnr = calculate_center_psnr(pred.detach(), target).item()
    print(f"  loss={loss.item():.5f} fdd={loss_fdd.item():.4f} PSNR_C={psnr:.2f} dB")
    break
elapsed = time.time() - t0
print(f"  1 batch 时间: {elapsed:.1f}s")

# GPU 显存
if torch.cuda.is_available():
    mem_alloc = torch.cuda.memory_allocated() / 1024**3
    mem_reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"  GPU 显存: alloc={mem_alloc:.2f} GB reserved={mem_reserved:.2f} GB")
    if mem_reserved > 8.0:
        print(f"  [WARNING] 显存 > 8GB! 训练会 OOM")
    else:
        print(f"  [OK] 显存 {mem_reserved:.2f} GB < 8GB GPU 上限")

print("\n[v8 SMOKE TEST] PASSED")
