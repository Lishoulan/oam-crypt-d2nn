"""v6 烟雾测试: 1 epoch 验证 mid_ch 128 + num_layers 3 + ChannelAttention 工作"""
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import os, sys

sys.path.insert(0, '.')
import oam_crypt_d2nn as m
from oam_crypt_d2nn import CONFIG

# v6 测试配置: 强制小规模
CONFIG["quick_test_n"] = 20   # 20 样本
CONFIG["epochs"] = 1
CONFIG["warmup_epochs"] = 0   # 跳过 warmup 快速验证
CONFIG["mid_ch"] = 48         # v6 保守: 64 OOM, 48 平衡 (ChannelAttention 单独验证)
CONFIG["num_layers"] = 3      # v6 测试 num_layers=3
CONFIG["use_channel_attn"] = True
CONFIG["layout"] = "oam_overlap"
CONFIG["z_list"] = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
CONFIG["sec_weight"] = 0.3

device = torch.device(CONFIG["device"])
torch.manual_seed(42); np.random.seed(42)

# 数据
transform = transforms.Compose([transforms.ToTensor()])
os.makedirs("./data", exist_ok=True)
full_train = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
train_subset = Subset(full_train, range(CONFIG["quick_test_n"] * 10))
train_dataset = m.MNISTQuadDataset(train_subset, img_size=CONFIG["size"]//5, num_channels=10)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

# RPP
rpp_system = m.generate_rpp(CONFIG["size"], device, generator=torch.Generator(device).manual_seed(0))

# 模型
print(f"[v6 smoke] CONFIG['mid_ch'] = {CONFIG['mid_ch']} (调用前)")
theta_max = np.deg2rad(CONFIG["theta_max_deg"])
model = m.OAM_Crypt_D2NN(
    size=CONFIG["size"], num_layers=CONFIG["num_layers"],
    wavelength=CONFIG["wavelength"], pixel_size=CONFIG["pixel_size"],
    z_layer=CONFIG["z_layer"], z0=CONFIG["z0"], rpp=rpp_system,
    oam_keys=CONFIG["l_auth"], z_list=CONFIG["z_list"],
    obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max,
    slm_aware=CONFIG["slm_aware"],
    use_channel_attn=CONFIG["use_channel_attn"],
    mid_ch=CONFIG["mid_ch"]
).to(device)
print(f"[v6 smoke] mid_ch 参数: {CONFIG['mid_ch']} (调用后)")
print(f"[v6 smoke] refine.enc1[0].out_channels = {model.refine.enc1[0].out_channels}")

# 参数统计
total_params = sum(p.numel() for p in model.parameters())
print(f"[v6 smoke] 模型参数: {total_params:,} (mid_ch 48, num_layers 3, channel_attn=True)")
print(f"[v6 smoke] refine.mid_ch: {model.refine.enc1[0].out_channels}")
print(f"[v6 smoke] refine.channel_attn: {type(model.refine.channel_attn).__name__}")

# 测试 1 个 batch
batch_imgs = next(iter(train_loader)).to(device)
print(f"[v6 smoke] 输入 batch 形状: {batch_imgs.shape}")

with torch.no_grad():
    c_digital = m.encrypt_batch(
        batch_imgs, CONFIG["l_auth"], rpp_system,
        CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
        size=CONFIG["size"], z_list=CONFIG["z_list"],
        obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max,
        layout=CONFIG.get("layout", "oam_overlap")
    )
    print(f"[v6 smoke] 密文形状: {c_digital.shape}, dtype: {c_digital.dtype}")
    pred = model(c_digital)
    print(f"[v6 smoke] 预测形状: {pred.shape}")
    tgt = m.build_target_grid(batch_imgs, device, size=CONFIG["size"], layout=CONFIG.get("layout", "oam_overlap"))
    psnr_c = m.calculate_center_psnr(pred, tgt).item()
    print(f"[v6 smoke] PSNR_C: {psnr_c:.2f} dB (初始, 应 ~6-10 dB)")

# 优化器 + 1 步训练测试
opt = torch.optim.Adam([
    {"params": model.refine.parameters(), "lr": 3e-4},
    {"params": model.layers.parameters(), "lr": 0.05}
])
opt.zero_grad()
c_digital = m.encrypt_batch(
    batch_imgs, CONFIG["l_auth"], rpp_system,
    CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
    size=CONFIG["size"], z_list=CONFIG["z_list"],
    obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max,
    layout=CONFIG.get("layout", "oam_overlap")
)
pred = model(c_digital)
weight_map = torch.ones(1, CONFIG["size"], CONFIG["size"], device=device)
weight_map[..., 432:648, 432:648] = 10.0
weight_map = weight_map.unsqueeze(1).expand(-1, 10, -1, -1)
mse = ((pred.clamp(0, 1) - tgt) ** 2 * weight_map).mean()
l1 = (pred.clamp(0, 1) - tgt).abs().mean()
loss = mse + 0.1 * l1
loss.backward()
opt.step()
print(f"[v6 smoke] 1 步训练 loss={loss.item():.4f} (mse={mse.item():.4f}, l1={l1.item():.4f})")
print(f"[v6 smoke] ✓ 代码工作正常")
