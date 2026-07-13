"""v7 SLM 加载仿真测试 - 测 10 通道 + Stage 1 2 通道 SLM 加载损耗"""
import os, sys, numpy as np, torch
sys.path.insert(0, 'f:/d2nn')
import oam_crypt_d2nn as m
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from font_config import setup_cjk
setup_cjk()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42); np.random.seed(42)

theta_max = np.deg2rad(m.CONFIG['theta_max_deg'])
rpp_system = m.generate_rpp(m.CONFIG['size'], device)

transform = transforms.Compose([transforms.ToTensor()])
full_test = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
mnist_test = Subset(full_test, range(50))

def test_slm_load(ckpt_path, num_channels, z_list, layout, label):
    print(f"\n{'='*80}\n[v7 SLM TEST] {label}\n  ckpt={ckpt_path}\n  n_channels={num_channels} z_list={z_list}\n{'='*80}", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"  loaded PSNR_C={ckpt.get('psnr_center', float('nan')):.2f} dB", flush=True)

    # 重建模型
    model = m.OAM_Crypt_D2NN(
        size=m.CONFIG['size'], num_layers=m.CONFIG['num_layers'],
        wavelength=m.CONFIG['wavelength'], pixel_size=m.CONFIG['pixel_size'],
        z_layer=m.CONFIG['z_layer'], z0=m.CONFIG['z0'], rpp=rpp_system,
        oam_keys=ckpt.get('l_auth') or m.CONFIG['l_auth'][:num_channels],
        z_list=z_list,
        obj_encoding=m.CONFIG['obj_encoding'], theta_max=theta_max,
        use_channel_attn=m.CONFIG['use_channel_attn'],
        mid_ch=m.CONFIG['mid_ch'],
        iterative_refine=m.CONFIG.get('iterative_refine', False),
        n_passes=m.CONFIG.get('n_passes', 3),
        oam_freq_filter=m.CONFIG.get('oam_freq_filter', True),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 数据
    test_dataset = m.MNISTQuadDataset(mnist_test, img_size=m.CONFIG['size']//5, num_channels=num_channels)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    sample = next(iter(test_loader)).to(device)

    with torch.no_grad():
        c_digital = m.encrypt_batch(
            sample, ckpt.get('l_auth') or m.CONFIG['l_auth'][:num_channels], rpp_system, m.CONFIG['z0'],
            m.CONFIG['wavelength'], m.CONFIG['pixel_size'], device,
            size=m.CONFIG['size'], z_list=z_list,
            obj_encoding=m.CONFIG['obj_encoding'], theta_max=theta_max,
            layout=layout
        )
        pred_digital = model(c_digital).clamp(0, 1)
        tgt = m.build_target_grid(sample, device, size=m.CONFIG['size'], layout=layout)
        psnr_digital = m.calculate_center_psnr(pred_digital, tgt).item()
        print(f"  数字仿真 PSNR_C: {psnr_digital:.2f} dB", flush=True)

        # SLM 加载流程
        U_dpe_complex = m.double_phase_encode(c_digital, device)[0]
        phase_slm = torch.angle(U_dpe_complex)
        gray_slm = ((phase_slm + np.pi) / (2 * np.pi) * 255).round().clamp(0, 255)
        gray_slm_np = gray_slm.cpu().numpy().astype(np.uint8)
        np.save(f"slm_hologram_v7_{label.replace(' ', '_')}.npy", gray_slm_np)

        phase_loaded = gray_slm.float() / 255.0 * 2 * np.pi - np.pi
        U_slm = torch.exp(1j * phase_loaded)
        c_slm = U_slm.unsqueeze(0)
        pred_slm = model(c_slm).clamp(0, 1)
        psnr_slm = m.calculate_center_psnr(pred_slm, tgt).item()
        slm_loss = psnr_digital - psnr_slm
        print(f"  SLM 仿真 PSNR_C: {psnr_slm:.2f} dB", flush=True)
        print(f"  *** SLM 加载损耗: {slm_loss:+.2f} dB ***", flush=True)

    return psnr_digital, psnr_slm

# Test 1: v7 final (10 通道, 13.88 dB)
p_d1, p_s1 = test_slm_load(
    ckpt_path="oam_crypt_v7_final.pth",
    num_channels=10,
    z_list=[0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
    layout="oam_overlap",
    label="v7_final_10ch_13.88dB"
)

# Test 2: v7 stage 1 (2 通道, 22.89 dB) - v7 真正突破
p_d2, p_s2 = test_slm_load(
    ckpt_path="oam_crypt_v7_stage1_best.pth",
    num_channels=2,
    z_list=[0.10, 0.55],
    layout="oam_overlap",
    label="v7_stage1_2ch_22.89dB"
)

# Test 3: v7 stage 2 (5 通道, 17.80 dB)
p_d3, p_s3 = test_slm_load(
    ckpt_path="oam_crypt_v7_stage2_best.pth",
    num_channels=5,
    z_list=[0.10, 0.20, 0.35, 0.45, 0.55],
    layout="oam_overlap",
    label="v7_stage2_5ch_17.80dB"
)

print(f"\n{'='*80}\n[v7 SLM 汇总]", flush=True)
print(f"  Stage 1 (2ch):  数字={p_d2:.2f} dB  SLM={p_s2:.2f} dB  损耗={p_d2-p_s2:+.2f} dB ⭐ 突破", flush=True)
print(f"  Stage 2 (5ch):  数字={p_d3:.2f} dB  SLM={p_s3:.2f} dB  损耗={p_d3-p_s3:+.2f} dB", flush=True)
print(f"  Stage 4 (10ch): 数字={p_d1:.2f} dB  SLM={p_s1:.2f} dB  损耗={p_d1-p_s1:+.2f} dB", flush=True)
print(f"{'='*80}\n", flush=True)
