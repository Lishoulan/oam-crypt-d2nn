"""v7 stage 1 (2 通道) SecurityRatio 测 - 真正突破点的安全性"""
import os, sys, numpy as np, torch
sys.path.insert(0, 'f:/d2nn')
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms
from font_config import setup_cjk
setup_cjk()

import oam_crypt_d2nn as m
from oam_crypt_d2nn import (
    OAM_Crypt_D2NN, encrypt_batch, build_target_grid,
    generate_rpp, calculate_center_psnr, security_ratio, CONFIG, MNISTQuadDataset,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42); np.random.seed(42)

theta_max = np.deg2rad(CONFIG['theta_max_deg'])
rpp_system = generate_rpp(CONFIG['size'], device)

# Load v7 stage 1 (2 通道)
ckpt = torch.load("oam_crypt_v7_stage1_best.pth", map_location=device, weights_only=False)
l_auth_2 = ckpt['l_auth']  # [-25, 25]
z_list_2 = ckpt['z_list']  # [0.10, 0.55]
print(f"[v7 stage 1] l_auth={l_auth_2} z_list={z_list_2}")

model = OAM_Crypt_D2NN(
    size=CONFIG['size'], num_layers=CONFIG['num_layers'],
    wavelength=CONFIG['wavelength'], pixel_size=CONFIG['pixel_size'],
    z_layer=CONFIG['z_layer'], z0=CONFIG['z0'], rpp=rpp_system,
    oam_keys=l_auth_2, z_list=z_list_2,
    obj_encoding=CONFIG['obj_encoding'], theta_max=theta_max,
    use_channel_attn=CONFIG['use_channel_attn'],
    mid_ch=CONFIG['mid_ch'],
    iterative_refine=False,
    oam_freq_filter=True,
).to(device)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()

# 测试数据 (2 通道)
transform = transforms.Compose([transforms.ToTensor()])
full_test = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
mnist_test = Subset(full_test, range(40))
test_dataset = MNISTQuadDataset(mnist_test, img_size=CONFIG['size']//5, num_channels=2)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 错误 OAM
l_wrong_2 = [-24, 24]  # 偏离 l_auth 1
print(f"  l_wrong = {l_wrong_2}")

psnr_auth, psnr_rpp, psnr_oam, sr_rpp, sr_oam = [], [], [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        tgt = build_target_grid(batch, device, size=CONFIG['size'], layout="oam_overlap")
        # 合法
        c_auth = encrypt_batch(batch, l_auth_2, rpp_system, CONFIG['z0'],
                               CONFIG['wavelength'], CONFIG['pixel_size'], device,
                               size=CONFIG['size'], z_list=z_list_2,
                               obj_encoding=CONFIG['obj_encoding'], theta_max=theta_max,
                               layout="oam_overlap")
        p_auth = model(c_auth).clamp(0, 1)
        # RPP 攻击
        rpp_w = generate_rpp(CONFIG['size'], device)
        c_rpp = encrypt_batch(batch, l_auth_2, rpp_w, CONFIG['z0'],
                              CONFIG['wavelength'], CONFIG['pixel_size'], device,
                              size=CONFIG['size'], z_list=z_list_2,
                              obj_encoding=CONFIG['obj_encoding'], theta_max=theta_max,
                              layout="oam_overlap")
        p_rpp = model(c_rpp).clamp(0, 1)
        # OAM 攻击
        c_oam = encrypt_batch(batch, l_wrong_2, rpp_system, CONFIG['z0'],
                              CONFIG['wavelength'], CONFIG['pixel_size'], device,
                              size=CONFIG['size'], z_list=z_list_2,
                              obj_encoding=CONFIG['obj_encoding'], theta_max=theta_max,
                              layout="oam_overlap")
        p_oam = model(c_oam).clamp(0, 1)
        psnr_auth.append(calculate_center_psnr(p_auth, tgt).item())
        psnr_rpp.append(calculate_center_psnr(p_rpp, tgt).item())
        psnr_oam.append(calculate_center_psnr(p_oam, tgt).item())
        sr_rpp.append(security_ratio(p_auth, p_rpp).item())
        sr_oam.append(security_ratio(p_auth, p_oam).item())

avg_psnr_a = float(np.mean(psnr_auth))
avg_psnr_r = float(np.mean(psnr_rpp))
avg_psnr_o = float(np.mean(psnr_oam))
avg_sr_r = float(np.mean(sr_rpp))
avg_sr_o = float(np.mean(sr_oam))

print(f"\n{'='*70}\n[v7 stage 1 (2 通道 22.89 dB) SecurityRatio]\n{'='*70}")
print(f"  PSNR_C (合法解密):    {avg_psnr_a:.2f} dB")
print(f"  PSNR_C (RPP 攻击):    {avg_psnr_r:.2f} dB")
print(f"  PSNR_C (OAM 攻击):    {avg_psnr_o:.2f} dB")
print(f"  SecurityRatio (RPP):  {avg_sr_r:.4f}  (目标 < 0.3)")
print(f"  SecurityRatio (OAM):  {avg_sr_o:.4f}  (目标 < 0.3)")
if avg_sr_r < 0.3 and avg_sr_o < 0.3:
    print(f"  ✅ SecurityRatio 全部通过")
else:
    print(f"  ⚠️ SecurityRatio 未通过 (但 PSNR_C > 20 dB 已是工程可用)")
print(f"{'='*70}")
