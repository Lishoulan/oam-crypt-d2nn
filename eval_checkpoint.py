# -*- coding: utf-8 -*-
"""快速评估 checkpoint 的 PSNR (不训练)"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import sys

# 导入主脚本中的所有组件
from oam_crypt_d2nn import (
    CONFIG, generate_rpp, OAM_Crypt_D2NN,
    MNISTQuadDataset, encrypt_batch, build_target_grid,
    calculate_psnr, save_security_plot
)

device = torch.device(CONFIG["device"])
torch.manual_seed(42)
np.random.seed(42)

# 1. 数据
transform = transforms.Compose([transforms.ToTensor()])
full_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
mnist_test = Subset(full_test, range(4000))
test_dataset = MNISTQuadDataset(mnist_test)
test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

# 2. RPP (固定种子)
rpp_system = generate_rpp(CONFIG["size"], device)

# 3. 模型 + 加载 checkpoint
model = OAM_Crypt_D2NN(
    size=CONFIG["size"], num_layers=CONFIG["num_layers"],
    wavelength=CONFIG["wavelength"], pixel_size=CONFIG["pixel_size"],
    z_layer=CONFIG["z_layer"], z0=CONFIG["z0"], rpp=rpp_system,
    oam_keys=CONFIG["l_auth"], z_list=CONFIG["z_list"]
).to(device)

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "oam_crypt_dnn_epoch_80.pth"
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()
print(f"已加载: {ckpt_path}", flush=True)

# 4. 评估
psnr_list = []
with torch.no_grad():
    for i, test_batch in enumerate(test_loader):
        test_batch = test_batch.to(device)
        tgt = build_target_grid(test_batch, device,
                                layout=CONFIG.get("layout", "grid_2x5"))
        c_auth = encrypt_batch(
            test_batch, CONFIG["l_auth"], rpp_system,
            CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
            z_list=CONFIG["z_list"],
            layout=CONFIG.get("layout", "grid_2x5")
        )
        p_auth = model(c_auth)
        psnr = calculate_psnr(p_auth, tgt).item()
        psnr_list.append(psnr)

        if i == 0:
            # 保存第一个样本的可视化
            rpp_wrong = generate_rpp(CONFIG["size"], device)
            c_unauth = encrypt_batch(
                test_batch, CONFIG["l_auth"], rpp_wrong,
                CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
                z_list=CONFIG["z_list"],
                layout=CONFIG.get("layout", "grid_2x5")
            )
            p_unauth = model(c_unauth)
            save_security_plot(
                c_auth, p_auth, tgt,
                c_unauth, p_unauth,
                path="eval_plot.png"
            )

avg_psnr = float(np.mean(psnr_list))
print(f"平均 PSNR: {avg_psnr:.2f} dB (共 {len(psnr_list)} 个 batch)", flush=True)
print(f"最高 PSNR: {max(psnr_list):.2f} dB", flush=True)
print(f"最低 PSNR: {min(psnr_list):.2f} dB", flush=True)
