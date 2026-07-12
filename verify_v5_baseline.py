"""v5 baseline 回归验证 - 加载 v5_oam_overlap_best_14.17dB.pth, 确认 v6 代码改动不破坏 v5 性能"""
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import os, sys

sys.path.insert(0, '.')
import oam_crypt_d2nn as m
from oam_crypt_d2nn import CONFIG

# v5 baseline 验证 (oam_overlap + 10cm 间距, 与 v5 训练一致)
CONFIG["layout"] = "oam_overlap"
CONFIG["z_list"] = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
# v5 训练用的是 mid_ch=64, num_layers=2, channel_attn=False
CONFIG["mid_ch"] = 64
CONFIG["num_layers"] = 2
CONFIG["use_channel_attn"] = False  # v5 没用 ChannelAttention
print(f"[v5 baseline 回归] mid_ch={CONFIG['mid_ch']}, num_layers={CONFIG['num_layers']}, channel_attn={CONFIG['use_channel_attn']}")

device = torch.device(CONFIG["device"])
torch.manual_seed(42); np.random.seed(42)

# 数据
transform = transforms.Compose([transforms.ToTensor()])
full_test = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
mnist_test = Subset(full_test, range(400))
test_dataset = m.MNISTQuadDataset(mnist_test, img_size=CONFIG["size"]//5, num_channels=10)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

rpp_system = m.generate_rpp(CONFIG["size"], device, generator=torch.Generator(device).manual_seed(0))

# 加载 v5 best
ckpt_path = "v5_oam_overlap_best_14.17dB.pth"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
print(f"\n[加载] {ckpt_path}: PSNR_C={ckpt.get('psnr_center', float('nan')):.2f}dB")

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

if "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)
model.eval()

# 评估 PSNR_C
psnr_center_list = []
with torch.no_grad():
    for i, batch_imgs in enumerate(test_loader):
        batch_imgs = batch_imgs.to(device)
        tgt = m.build_target_grid(batch_imgs, device, size=CONFIG["size"], layout=CONFIG["layout"])
        c_auth = m.encrypt_batch(
            batch_imgs, CONFIG["l_auth"], rpp_system,
            CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
            size=CONFIG["size"], z_list=CONFIG["z_list"],
            obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max,
            layout=CONFIG["layout"]
        )
        p_auth = model(c_auth)
        psnr_c = m.calculate_center_psnr(p_auth, tgt).item()
        psnr_center_list.append(psnr_c)

print(f"\n=== v5 baseline 验证 (oam_overlap + 10cm 间距) ===")
print(f"  测试样本数: {len(psnr_center_list)}")
print(f"  平均 PSNR_C: {np.mean(psnr_center_list):.2f} dB (v5 训练时 14.17 dB)")
print(f"  最高 PSNR_C: {max(psnr_center_list):.2f} dB")
print(f"  最低 PSNR_C: {min(psnr_center_list):.2f} dB")
print(f"  {'✓ 验证通过' if np.mean(psnr_center_list) >= 10 else '⚠ 退化'}")
