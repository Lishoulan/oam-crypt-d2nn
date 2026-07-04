import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 阶段1：基础参数配置
# ==========================================
CONFIG = {
    "size": 128,              # 系统计算尺寸 (128x128)
    "wavelength": 633e-9,     # 光波长 633 nm
    "pixel_size": 8e-6,       # 像素大小 8 um
    "z0": 0.1,                # 物面到全息面的自由空间传播距离 (0.1 m)
    "z_layer": 0.02,          # D2NN 相位层之间的传播距离 (0.02 m)
    "l_list": [-3, -1, 1, 3], # 4个OAM通道的拓扑荷数
    "batch_size": 64,         # 批大小
    "epochs": 200,            # 训练轮次
    "lr": 1e-3,               # 初始学习率
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

print(f"当前运行设备: {CONFIG['device']}")


# ==========================================
# 阶段2：核心物理算子与层定义
# ==========================================

def propagate_asm(U_in, z, wavelength, pixel_size, device):
    """
    基于角谱法(ASM)的可微自由空间传播算子
    支持输入形状 (B, H, W) 或 (H, W)
    """
    if len(U_in.shape) == 2:
        U_in = U_in.unsqueeze(0)
        is_batched = False
    else:
        is_batched = True

    B, H, W = U_in.shape
    # 构建频域网格
    fx = torch.fft.fftfreq(W, d=pixel_size, device=device)
    fy = torch.fft.fftfreq(H, d=pixel_size, device=device)
    f_y, f_x = torch.meshgrid(fy, fx, indexing='ij')

    # 计算传播传递函数
    k = 2 * np.pi / wavelength
    term = 1.0 - (wavelength * f_x) ** 2 - (wavelength * f_y) ** 2
    mask = (term >= 0).float()
    pz = torch.sqrt(torch.clamp(term, min=0.0))
    H_kernel = torch.exp(1j * k * z * pz) * mask

    # 傅里叶变换与频域滤波
    U_fft = torch.fft.fft2(U_in)
    U_out = torch.fft.ifft2(U_fft * H_kernel)

    if not is_batched:
        U_out = U_out.squeeze(0)
    return U_out


def generate_oam_phase(size, l, device):
    """
    生成特定拓扑荷数 l 的 OAM 螺旋相位矩阵
    """
    y = torch.linspace(-size // 2, size // 2 - 1, size, device=device)
    x = torch.linspace(-size // 2, size // 2 - 1, size, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    theta = torch.atan2(yy, xx)
    return torch.exp(1j * l * theta)


class DiffractiveLayer(nn.Module):
    """
    可训练纯相位调制衍射层
    """
    def __init__(self, size):
        super().__init__()
        # 随机初始化相位值在 [-pi, pi] 之间
        self.phase = nn.Parameter(torch.empty(size, size).uniform_(-np.pi, np.pi))

    def forward(self, U):
        # 调制输入复振幅
        modulation = torch.exp(1j * self.phase)
        return U * modulation


# ==========================================
# 阶段3：数据集构建与合成
# ==========================================

class MNISTQuadDataset(Dataset):
    """
    将 MNIST 手写体图像进行分组打包。
    每次读取返回 4 张缩放到 64x64 的图像，用于后续拼装 2x2 网格。
    """
    def __init__(self, mnist_dataset):
        self.mnist = mnist_dataset
        self.num_samples = len(self.mnist) // 4

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        imgs = []
        for i in range(4):
            img, _ = self.mnist[idx * 4 + i]
            # 缩放至 64x64
            img_64 = transforms.functional.resize(img, [64, 64])
            imgs.append(img_64[0])  # 提取通道，形状为 (64, 64)
        return torch.stack(imgs, dim=0)  # 返回 (4, 64, 64)


def generate_batch_hologram(batch_imgs, l_list, z0, wavelength, pixel_size, device):
    """
    在 GPU 上并行生成「OAM复用全息图」输入以及对应的「2x2空间排布」标签。
    - 输入：batch_imgs 形状为 (B, 4, 64, 64)
    - 输出：
      U_input: 复数张量 (B, 128, 128)，代表叠加后的复用全息面
      target: 实数张量 (B, 128, 128)，代表目标重建平面
    """
    B = batch_imgs.shape[0]

    # 1. 构建 2x2 拼装的目标标签图
    target = torch.zeros(B, 128, 128, device=device)
    target[:, 0:64, 0:64] = batch_imgs[:, 0]      # 左上 -> 通道0
    target[:, 0:64, 64:128] = batch_imgs[:, 1]     # 右上 -> 通道1
    target[:, 64:128, 0:64] = batch_imgs[:, 2]     # 左下 -> 通道2
    target[:, 64:128, 64:128] = batch_imgs[:, 3]   # 右下 -> 通道3

    # 2. 生成多路物场的 OAM 调制传播与复用叠加
    U_input = torch.zeros(B, 128, 128, dtype=torch.complex64, device=device)

    for i, l in enumerate(l_list):
        # 将各通道 64x64 的原始物图居中填充至 128x128 独立物面
        img_pad = torch.zeros(B, 128, 128, device=device)
        img_pad[:, 32:96, 32:96] = batch_imgs[:, i]

        # 物理上光场的复振幅与其强度的平方根成正比
        amp = torch.sqrt(img_pad)
        U_obj = amp.to(torch.complex64)

        # 正向传播到全息面
        U_holo = propagate_asm(U_obj, z0, wavelength, pixel_size, device)

        # 乘上该通道的 OAM 相位
        oam_phase = generate_oam_phase(128, l, device)
        U_input += U_holo * oam_phase

    return U_input, target


# ==========================================
# 阶段4：网络架构设计
# ==========================================

class OAM_Holo_DNN(nn.Module):
    """
    一体化解复用 + 重建的衍射神经网络 (D2NN)
    """
    def __init__(self, size=128, num_layers=4, wavelength=633e-9, pixel_size=8e-6, z_layer=0.02):
        super().__init__()
        self.layers = nn.ModuleList([DiffractiveLayer(size) for _ in range(num_layers)])
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.z_layer = z_layer

    def forward(self, U):
        device = U.device
        for layer in self.layers:
            # 经可训练相位层调制
            U = layer(U)
            # 传播固定自由空间距离到下一层或输出面
            U = propagate_asm(U, self.z_layer, self.wavelength, self.pixel_size, device)

        # 探测器接收面记录光强
        intensity = torch.abs(U) ** 2
        return intensity


# ==========================================
# 阶段5：验证指标与辅助函数
# ==========================================

def calculate_psnr(pred, target):
    """
    计算峰值信噪比 (PSNR)
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    max_val = 1.0  # 已归一化到 [0, 1]
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def save_and_plot_comparison(holo_input, pred, target, epoch=None, path="reconstruction.png"):
    """
    绘制测试样本对比图：输入复用全息强度/相位、4路重建结果、4路原始标签
    """
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))

    # 全息面复振幅展现
    holo_amp = torch.abs(holo_input[0]).cpu().numpy()
    holo_phase = torch.angle(holo_input[0]).cpu().numpy()

    axes[0, 0].imshow(holo_amp, cmap='gray')
    axes[0, 0].set_title("Holo Amplitude")
    axes[0, 1].imshow(holo_phase, cmap='hsv')
    axes[0, 1].set_title("Holo Phase")

    # 隐藏首行多余子图
    for col in range(2, 4):
        axes[0, col].axis('off')

    # 分割重建图像象限
    rec_quads = [
        pred[0, 0:64, 0:64], pred[0, 0:64, 64:128],
        pred[0, 64:128, 0:64], pred[0, 64:128, 64:128]
    ]
    # 分割标签象限
    tgt_quads = [
        target[0, 0:64, 0:64], target[0, 0:64, 64:128],
        target[0, 64:128, 0:64], target[0, 64:128, 64:128]
    ]

    for i in range(4):
        axes[1, i].imshow(rec_quads[i].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f"Reconstructed Ch {i}")

        axes[2, i].imshow(tgt_quads[i].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[2, i].set_title(f"Target Ch {i}")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ==========================================
# 阶段6：主训练与测试流程
# ==========================================

if __name__ == "__main__":
    device = torch.device(CONFIG["device"])

    # 1. 载入并预处理 MNIST 数据
    transform = transforms.Compose([transforms.ToTensor()])

    # 自动下载官方数据集
    os.makedirs("./data", exist_ok=True)
    full_mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    full_mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 为加速数值仿真运行，选取训练集前8000张（可生成2000组样本），测试集前1600张（可生成400组样本）
    # 若需使用完整数据集，只需将 Subset 替换为原始数据集即可
    mnist_train_sub = Subset(full_mnist_train, range(8000))
    mnist_test_sub = Subset(full_mnist_test, range(1600))

    train_dataset = MNISTQuadDataset(mnist_train_sub)
    test_dataset = MNISTQuadDataset(mnist_test_sub)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    # 2. 实例化网络与优化器
    model = OAM_Holo_DNN(
        size=CONFIG["size"],
        num_layers=4,
        wavelength=CONFIG["wavelength"],
        pixel_size=CONFIG["pixel_size"],
        z_layer=CONFIG["z_layer"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    # 采用余弦退火策略调整学习率
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    criterion_mse = nn.MSELoss()

    print("开始训练物理衍射网络...")

    # 3. 训练循环
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        epoch_loss = 0.0

        for batch_imgs in train_loader:
            batch_imgs = batch_imgs.to(device)
            # 在 GPU 上并行合成全息图并生成目标大图
            U_input, target = generate_batch_hologram(
                batch_imgs, CONFIG["l_list"], CONFIG["z0"],
                CONFIG["wavelength"], CONFIG["pixel_size"], device
            )

            optimizer.zero_grad()
            pred = model(U_input)

            # 主损失：预测重建面与目标排布的 MSE
            loss_mse = criterion_mse(pred, target)

            # 串扰惩罚：抑制模式泄漏。计算重建象限 i 与非对应象限目标值 j 的像素点乘积
            loss_crosstalk = 0.0
            rec_quads = [pred[:, 0:64, 0:64], pred[:, 0:64, 64:128], pred[:, 64:128, 0:64], pred[:, 64:128, 64:128]]
            tgt_quads = [target[:, 0:64, 0:64], target[:, 0:64, 64:128], target[:, 64:128, 0:64], target[:, 64:128, 64:128]]

            for i in range(4):
                for j in range(4):
                    if i != j:
                        # 惩罚重构区 i 中泄漏的重构区 j 信息
                        loss_crosstalk += torch.mean(rec_quads[i] * tgt_quads[j])

            total_loss = loss_mse + 0.1 * loss_crosstalk
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item() * batch_imgs.size(0)

        scheduler.step()
        epoch_loss /= len(train_loader.dataset)

        # 4. 验证与日志输出
        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            test_psnr_list = []

            with torch.no_grad():
                for test_batch_imgs in test_loader:
                    test_batch_imgs = test_batch_imgs.to(device)
                    U_input_t, target_t = generate_batch_hologram(
                        test_batch_imgs, CONFIG["l_list"], CONFIG["z0"],
                        CONFIG["wavelength"], CONFIG["pixel_size"], device
                    )
                    pred_t = model(U_input_t)

                    psnr = calculate_psnr(pred_t, target_t)
                    test_psnr_list.append(psnr.item())

            avg_psnr = np.mean(test_psnr_list)
            print(f"Epoch [{epoch}/{CONFIG['epochs']}] | 训练 Loss: {epoch_loss:.6f} | 测试集平均 PSNR: {avg_psnr:.2f} dB")

            # 阶段性保存模型参数及生成效果图
            torch.save(model.state_dict(), f"oam_holo_dnn_epoch_{epoch}.pth")

            # 取第一组数据进行可视化绘图
            save_and_plot_comparison(
                U_input_t, pred_t, target_t,
                epoch=epoch, path=f"epoch_{epoch}_reconstruction.png"
            )

    print("训练结束。最终对比图已保存为 final_reconstruction.png")
    # 生成最终图
    save_and_plot_comparison(U_input_t, pred_t, target_t, path="final_reconstruction.png")
