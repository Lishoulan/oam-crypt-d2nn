# -*- coding: utf-8 -*-
"""诊断 v8 ckpt 加载是否正确"""
import sys
import numpy as np
import torch
sys.path.insert(0, '.')
from oam_crypt_d2nn import (
    CONFIG, OAM_Crypt_D2NN, generate_rpp, encrypt_batch, build_target_grid,
    calculate_center_psnr, MNISTQuadDataset
)
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

device = torch.device(CONFIG['device'])
torch.manual_seed(42)

l_auth = [-25, 25]
z_list = [0.10, 0.55]
rpp = generate_rpp(CONFIG['size'], device)

# 与 train_one_stage 一致
model = OAM_Crypt_D2NN(
    size=CONFIG['size'], num_layers=CONFIG['num_layers'],
    wavelength=CONFIG['wavelength'], pixel_size=CONFIG['pixel_size'],
    z_layer=CONFIG['z_layer'], z0=CONFIG['z0'], rpp=rpp,
    oam_keys=l_auth, z_list=z_list,
    obj_encoding=CONFIG['obj_encoding'],
    theta_max=np.deg2rad(CONFIG['theta_max_deg']) if CONFIG.get('theta_max_deg') else None,
    slm_aware=CONFIG['slm_aware'],
    use_channel_attn=True, mid_ch=CONFIG['mid_ch'],
    iterative_refine=False,
    oam_freq_filter=True,
    use_polar_conv=True,
    polar_n_r=32, polar_n_theta=96,
).to(device)
ckpt = torch.load("oam_crypt_v8_stage1_best.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

print(f"[CKPT] n_ch={ckpt['n_channels']} l_auth={ckpt['l_auth']} z_list={ckpt['z_list']}")
print(f"[CKPT] best_psnr={ckpt['psnr_center']:.2f}")

# 检查第一层 phase 范数
for i, layer in enumerate(model.layers):
    print(f"  layer {i}: phase std={layer.phase.std().item():.3f} amp mean={torch.sigmoid(layer.amp_logit).mean().item():.3f}")

# 跑单个 batch, 看 PSNR
transform = transforms.Compose([transforms.ToTensor()])
full_test = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
mnist_sub = Subset(full_test, range(0, 200))
ds = MNISTQuadDataset(mnist_sub, img_size=CONFIG['size']//5, num_channels=2)
loader = DataLoader(ds, batch_size=1, shuffle=False)

psnrs = []
for i, batch_imgs in enumerate(loader):
    if i >= 5: break
    batch_imgs = batch_imgs.to(device)
    target = build_target_grid(batch_imgs, device, size=CONFIG['size'], layout='oam_overlap')
    cipher = encrypt_batch(
        batch_imgs, l_auth, rpp,
        CONFIG['z0'], CONFIG['wavelength'], CONFIG['pixel_size'], device,
        size=CONFIG['size'], z_list=z_list,
        obj_encoding=CONFIG['obj_encoding'],
        theta_max=np.deg2rad(CONFIG['theta_max_deg']) if CONFIG.get('theta_max_deg') else None,
        layout='oam_overlap'
    )
    with torch.no_grad():
        pred = model(cipher)
    psnr = calculate_center_psnr(pred, target).item()
    print(f"  样本 {i}: pred range [{pred.min().item():.3f}, {pred.max().item():.3f}], PSNR_C={psnr:.2f} dB")
    psnrs.append(psnr)
print(f"  Mean PSNR_C (前 5): {np.mean(psnrs):.2f} dB")
