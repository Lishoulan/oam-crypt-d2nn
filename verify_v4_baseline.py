"""v4 baseline 验证 - 切换回 grid_2x5 布局 + 5cm 间距,加载 v4_baseline_29.79dB.pth"""
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import os, sys

sys.path.insert(0, '.')
import oam_crypt_d2nn as m
from oam_crypt_d2nn import CONFIG

# 临时覆盖 CONFIG (v4 baseline 设置)
print("[v4 baseline 验证] 临时切换 CONFIG 为 grid_2x5 + 5cm 间距")
CONFIG["layout"] = "grid_2x5"
CONFIG["z_list"] = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]  # 5cm 间距
print(f"  layout = {CONFIG['layout']}")
print(f"  z_list = {CONFIG['z_list']}")

device = torch.device(CONFIG["device"])
torch.manual_seed(42); np.random.seed(42)

# 数据
transform = transforms.Compose([transforms.ToTensor()])
full_test = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
mnist_test = Subset(full_test, range(400))
test_dataset = m.MNISTQuadDataset(mnist_test, img_size=CONFIG["size"]//5, num_channels=10)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

# RPP (固定种子)
rpp_system = m.generate_rpp(CONFIG["size"], device, generator=torch.Generator(device).manual_seed(0))

# 加载 v4 baseline
ckpt_path = "v4_baseline_29.79dB.pth"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
print(f"\n[加载] {ckpt_path}: PSNR_C={ckpt.get('psnr_center', float('nan')):.2f}dB")

theta_max = np.deg2rad(CONFIG["theta_max_deg"])
model = m.OAM_Crypt_D2NN(
    size=CONFIG["size"], num_layers=CONFIG["num_layers"],
    wavelength=CONFIG["wavelength"], pixel_size=CONFIG["pixel_size"],
    z_layer=CONFIG["z_layer"], z0=CONFIG["z0"], rpp=rpp_system,
    oam_keys=CONFIG["l_auth"], z_list=CONFIG["z_list"],
    obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max,
    slm_aware=CONFIG["slm_aware"]
).to(device)

if "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)
model.eval()

# 评估 PSNR_C
psnr_center_list = []
psnr_full_list = []
with torch.no_grad():
    for i, batch_imgs in enumerate(test_loader):
        batch_imgs = batch_imgs.to(device)
        tgt = m.build_target_grid(batch_imgs, device, size=CONFIG["size"],
                                  layout=CONFIG.get("layout", "grid_2x5"))
        c_auth = m.encrypt_batch(
            batch_imgs, CONFIG["l_auth"], rpp_system,
            CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
            size=CONFIG["size"], z_list=CONFIG["z_list"],
            obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max,
            layout=CONFIG.get("layout", "grid_2x5")
        )
        p_auth = model(c_auth)
        psnr_c = m.calculate_center_psnr(p_auth, tgt).item()
        psnr_full = m.calculate_psnr(p_auth, tgt).item()
        psnr_center_list.append(psnr_c)
        psnr_full_list.append(psnr_full)

print(f"\n=== v4 baseline 验证 (grid_2x5 + 5cm 间距) ===")
print(f"  测试样本数: {len(psnr_center_list)}")
print(f"  平均 PSNR_C: {np.mean(psnr_center_list):.2f} dB (v4 训练时 29.79 dB)")
print(f"  平均 PSNR (全图): {np.mean(psnr_full_list):.2f} dB")
print(f"  最高 PSNR_C: {max(psnr_center_list):.2f} dB")
print(f"  最低 PSNR_C: {min(psnr_center_list):.2f} dB")
print(f"  ✓ 验证通过 (≥ 25 dB)" if np.mean(psnr_center_list) >= 25 else f"  ⚠ PSNR_C < 25 dB, baseline 退化")

# 测试完切回 oam_overlap
CONFIG["layout"] = "oam_overlap"
CONFIG["z_list"] = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
print(f"\n[恢复] layout=oam_overlap, z_list=10cm 间距")
