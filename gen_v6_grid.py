"""v6 拼图生成: 调用 viz_decrypted_grid 输出 2x5 网格图 (v6 最佳 14.65 dB)"""
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import os, sys

sys.path.insert(0, '.')
import oam_crypt_d2nn as m
from oam_crypt_d2nn import CONFIG

device = torch.device(CONFIG["device"])
torch.manual_seed(42); np.random.seed(42)

# 数据
transform = transforms.Compose([transforms.ToTensor()])
full_test = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
mnist_test = Subset(full_test, range(50))
test_dataset = m.MNISTQuadDataset(mnist_test, img_size=CONFIG["size"]//5, num_channels=10)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
sample = next(iter(test_loader)).to(device)

# RPP
rpp_system = m.generate_rpp(CONFIG["size"], device, generator=torch.Generator(device).manual_seed(0))

# 加载 v6 best (注意 v6 用 mid_ch=48, num_layers=3, channel_attn=True)
ckpt_path = "v6_oam_overlap_v2_best_14.65dB.pth"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
print(f"[加载] {ckpt_path}: PSNR_C={ckpt.get('psnr_center', float('nan')):.2f}dB")

theta_max = np.deg2rad(CONFIG["theta_max_deg"])
model = m.OAM_Crypt_D2NN(
    size=CONFIG["size"], num_layers=CONFIG["num_layers"],
    wavelength=CONFIG["wavelength"], pixel_size=CONFIG["pixel_size"],
    z_layer=CONFIG["z_layer"], z0=CONFIG["z0"], rpp=rpp_system,
    oam_keys=CONFIG["l_auth"], z_list=CONFIG["z_list"],
    obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max,
    slm_aware=CONFIG["slm_aware"],
    use_channel_attn=CONFIG.get("use_channel_attn", True),
    mid_ch=CONFIG["mid_ch"]
).to(device)
if "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)
model.eval()

# 加密 + 解密
with torch.no_grad():
    c_digital = m.encrypt_batch(
        sample, CONFIG["l_auth"], rpp_system,
        CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
        size=CONFIG["size"], z_list=CONFIG["z_list"],
        obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max,
        layout=CONFIG.get("layout", "oam_overlap")
    )
    pred = model(c_digital).clamp(0, 1)
    tgt = m.build_target_grid(sample, device, size=CONFIG["size"],
                              layout=CONFIG.get("layout", "oam_overlap"))
    psnr_c = m.calculate_center_psnr(pred, tgt).item()
    print(f"\n数字仿真 PSNR_C: {psnr_c:.2f} dB")

# 生成 v6 解密拼图
m.viz_decrypted_grid(pred, save_path="decrypted_grid_2x5_v6.png")
m.viz_decrypted_grid(tgt.unsqueeze(0) if tgt.dim()==3 else tgt, save_path="target_grid_2x5_v6.png")

print(f"\n[完成] 拼图已保存: decrypted_grid_2x5_v6.png, target_grid_2x5_v6.png")
