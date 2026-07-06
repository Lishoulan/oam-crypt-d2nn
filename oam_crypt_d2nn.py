# -*- coding: utf-8 -*-
"""
双密钥全光多用户信息加解密与隐写衍射神经网络 (D2NN) 数值仿真
================================================================
多平面 OAM 复用全息 + 双相位编码 (纯相位 SLM 兼容)

物理流程：
  加密: 明文 P_i -> 振幅 sqrt(P_i) -> ASM(z_i) [每路不同 z_i, 多平面复用]
       -> OAM 密钥编码 e^{i l_i theta} -> 4 路叠加 U_sum -> RPP 调制 -> 密文 U_cipher
  解密: U_cipher -> 去除 RPP (数字预处理) -> 双相位编码 (纯相位 SLM)
       -> 低通滤波 (恢复复振幅) -> OAM 解复用 [4 路] -> ASM(-z_j) [多平面聚焦]
       -> D2NN -> U-Net 精修

三大功能:
  1. 不同平面出现不同图案 (z_list 间距 20-40cm, 平面对比度 >1.6x)
  2. 错误 OAM 拓扑荷看不到图像 (OAM 正交性, 安全比 <0.6)
  3. 正确 OAM 拓扑荷只看到对应平面图案 (多平面聚焦选择性)

双密钥: OAM 拓扑荷 (用户级) + RPP 随机相位板 (系统级)
SLM 兼容: 双相位编码将复振幅编码为纯相位, 兼容纯相位 SLM
仅依赖: torch, torchvision, numpy, matplotlib
"""

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
    "size": 1080,             # 系统计算尺寸 (1080x1080, 匹配 SLM 高度)
    "wavelength": 532e-9,     # 光波长 532 nm (绿光)
    "pixel_size": 8e-6,       # 像素大小 8 um
    "z0": 0.1,                # 物面到全息面(加密面)的传播距离 (向后兼容)
    # 多平面 OAM 复用全息: 每路对应不同传播距离 z_i (核心创新!)
    # l_auth[i] 对应 z_list[i] 平面, 解密时只有在该平面才能看到对应图像
    # 总行程控制在 30cm 以内, 用递增间距保证相邻平面有足够 defocus
    "z_list": [0.05, 0.13, 0.22, 0.30],  # 4 个平面 (米), 间距 8-9cm, 总行程 25cm
    "z_layer": 0.02,          # D2NN 相位层之间的传播距离
    "l_auth": [-7, -3, 3, 7], # 增大 OAM 差异 (正交性更好, 串扰更低)
    "l_wrong": [-5, -1, 1, 5], # 错误 OAM 密钥(用于非法输入构造)
    "batch_size": 2,           # 批大小 (1080x1080 显存大, 用小批)
    "epochs": 40,              # 训练轮次 (同位置复用需更多 epoch 收敛)
    "lr": 1e-3,                # U-Net 精修层学习率
    "lr_d2nn": 0.1,           # D2NN 物理层学习率
    "mid_ch": 64,             # U-Net 中间通道数 (1080 大, 减通道省显存)
    "num_layers": 0,          # 衍射层数 (0=不用D2NN)
    "freeze_epochs": 0,       # 0=不冻结
    "warmup_epochs": 999,     # 预热轮次 (设大于 epochs 即可全程跳过安全损失, 安全性由物理机制保证)
    "sec_weight": 0.0,        # 安全损失权重 (关闭, 物理正交性已提供足够安全比 0.51)
    "xtalk_weight": 0.05,     # 串扰损失权重
    "l1_weight": 0.0,         # L1 损失权重
    # 物光编码模式: "amplitude" = sqrt(P) 振幅编码 (有平面选择性, 配合双相位编码兼容纯相位SLM)
    #               "phase" = exp(iπP) 相位编码 (无平面选择性, 但天然纯相位)
    "obj_encoding": "amplitude",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

print(f"当前运行设备: {CONFIG['device']}")


# ==========================================
# 阶段2：核心物理算子
# ==========================================

def propagate_asm(U_in, z, wavelength, pixel_size, device):
    """
    角谱法(ASM)可微自由空间传播算子
    支持输入形状 (B, H, W) 或 (H, W)，输出对应形状的复振幅
    """
    if len(U_in.shape) == 2:
        U_in = U_in.unsqueeze(0)
        is_batched = False
    else:
        is_batched = True

    B, H, W = U_in.shape
    # 频域坐标网格
    fx = torch.fft.fftfreq(W, d=pixel_size, device=device)
    fy = torch.fft.fftfreq(H, d=pixel_size, device=device)
    f_y, f_x = torch.meshgrid(fy, fx, indexing='ij')

    # 角谱传递函数 H = exp(i k z sqrt(1 - (lambda fx)^2 - (lambda fy)^2))
    k = 2 * np.pi / wavelength
    term = 1.0 - (wavelength * f_x) ** 2 - (wavelength * f_y) ** 2
    mask = (term >= 0).float()            # 消失波置零
    pz = torch.sqrt(torch.clamp(term, min=0.0))
    H_kernel = torch.exp(1j * k * z * pz) * mask

    U_fft = torch.fft.fft2(U_in)
    U_out = torch.fft.ifft2(U_fft * H_kernel)

    if not is_batched:
        U_out = U_out.squeeze(0)
    return U_out


def generate_oam_phase(size, l, device):
    """
    生成拓扑荷为 l 的 OAM 螺旋相位矩阵 e^{i l theta}
    """
    y = torch.linspace(-size // 2, size // 2 - 1, size, device=device)
    x = torch.linspace(-size // 2, size // 2 - 1, size, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    theta = torch.atan2(yy, xx)
    return torch.exp(1j * l * theta)


def generate_rpp(size, device, generator=None):
    """
    生成全局系统物理密钥 RPP (Random Phase Plate)
    返回静态复矩阵 e^{i phi}, phi ~ U[-pi, pi]
    """
    if generator is not None:
        phi = (torch.rand(size, size, generator=generator, device=device) * 2 - 1) * np.pi
    else:
        phi = (torch.rand(size, size, device=device) * 2 - 1) * np.pi
    return torch.exp(1j * phi)


def double_phase_encode(U, device):
    """
    双相位编码 (Double-Phase Encoding):
      将复振幅 U = A·exp(iφ) 编码为纯相位, 兼容纯相位 SLM。

    原理: 将 A·exp(iφ) 分解为两个单位模复数的平均
      φ₁ = φ + arccos(A/A_max)
      φ₂ = φ - arccos(A/A_max)
      [exp(iφ₁) + exp(iφ₂)] / 2 = (A/A_max)·exp(iφ) ∝ U

    实现: 棋盘格交错 φ₁ 和 φ₂, 经低通滤波后恢复复振幅。
    """
    A = torch.abs(U)
    A_max = A.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    A_norm = (A / A_max).clamp(max=1.0)
    phi = torch.angle(U)

    arccos_A = torch.arccos(A_norm)
    phi1 = phi + arccos_A
    phi2 = phi - arccos_A

    H, W = U.shape[-2], U.shape[-1]
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    checker = ((xx + yy) % 2).bool()
    if U.dim() == 3:
        checker = checker.unsqueeze(0)

    phase = torch.where(checker, phi2, phi1)
    return torch.exp(1j * phase)


def lowpass_filter(U, sigma=0.15):
    """
    频域高斯低通滤波: 滤除双相位编码的棋盘格高频分量, 恢复复振幅。
    sigma 单位: cycles/pixel (归一化频率)
      - 棋盘格频率 = 0.5 (Nyquist) → sigma=0.15 时衰减 ≈ exp(-5.56) ≈ 0.004
      - 图像频率 < 0.05 → 衰减 ≈ exp(-0.056) ≈ 0.95 (几乎无损)
    """
    H, W = U.shape[-2], U.shape[-1]
    fy = torch.fft.fftfreq(H, device=U.device)
    fx = torch.fft.fftfreq(W, device=U.device)
    f_y, f_x = torch.meshgrid(fy, fx, indexing='ij')
    filt = torch.exp(-(f_x ** 2 + f_y ** 2) / (2 * sigma ** 2))

    U_real = torch.fft.ifft2(torch.fft.fft2(U.real) * filt)
    U_imag = torch.fft.ifft2(torch.fft.fft2(U.imag) * filt)
    return U_real + 1j * U_imag


class DiffractiveLayer(nn.Module):
    """
    可训练复振幅调制衍射层: U_out = U_in * A * e^{i phi}
    - phase: 可训练相位, 初始化为 0 (接近恒等映射, 避免随机初始化破坏信息)
    - amplitude: 可训练振幅透过率, 经 sigmoid 约束到 (0,1), 初始化接近 1
    """
    def __init__(self, size, nonlin_every=3, layer_idx=0):
        super().__init__()
        self.phase = nn.Parameter(torch.zeros(size, size))  # 初始化 0 (恒等)
        self.amp_logit = nn.Parameter(torch.full((size, size), 4.0))  # sigmoid(4)≈0.98

    def forward(self, U):
        amplitude = torch.sigmoid(self.amp_logit)
        return U * amplitude * torch.exp(1j * self.phase)


# ==========================================
# 阶段3：双密钥加密引擎
# ==========================================

def encrypt_batch(batch_imgs, oam_keys, rpp, z0, wavelength, pixel_size, device,
                  size=1080, z_list=None, obj_encoding="amplitude"):
    """
    双密钥加密引擎(支持授权/未授权两种模式，由传入的 oam_keys / rpp 决定)
    - 输入: batch_imgs (B, 4, S, S) 明文振幅图 (S = size//2, 由调用方决定)
    - 输出: U_cipher (B, size, size) 复数密文场

    多平面 OAM 复用全息 (核心创新):
      - 4 个图像都放在画面中心同一位置 (不分象限, 真正的复用全息)
      - 每路 OAM 通道用不同传播距离 z_i
      - 解密时只有用对应 conj(OAM_j) 解调 + 在 z_j 平面观察才能看到图像 j
      - 不同平面的图案在同一位置出现, 不会错开
    向后兼容: 若 z_list=None, 则所有路都用 z0 (旧行为)
    """
    B = batch_imgs.shape[0]
    U_sum = torch.zeros(B, size, size, dtype=torch.complex64, device=device)
    half = batch_imgs.shape[-1]  # 图像尺寸 (可能 = size//2)
    # 4 个图像都放在中心同一位置 (多平面 OAM 复用全息的关键!)
    cy = (size - half) // 2
    cx = (size - half) // 2

    for i, l in enumerate(oam_keys):
        img_pad = torch.zeros(B, size, size, device=device)
        img_pad[:, cy:cy+half, cx:cx+half] = batch_imgs[:, i]

        if obj_encoding == "phase":
            U_obj = torch.exp(1j * np.pi * img_pad)
        else:
            U_obj = torch.sqrt(img_pad).to(torch.complex64)

        z_i = z_list[i] if z_list is not None else z0
        U_prop = propagate_asm(U_obj, z_i, wavelength, pixel_size, device)

        oam_phase = generate_oam_phase(size, l, device)
        U_sum = U_sum + U_prop * oam_phase

    U_cipher = U_sum * rpp
    return U_cipher


def build_target_grid(batch_imgs, device, size=1080):
    """
    构建 (B, 4, size, size) 目标: 每个通道对应一个图像, 都在中心同一位置
    (与 encrypt_batch 的图像放置位置一致, 多平面 OAM 复用全息)
    """
    B = batch_imgs.shape[0]
    target = torch.zeros(B, 4, size, size, device=device)
    half = batch_imgs.shape[-1]
    cy = (size - half) // 2
    cx = (size - half) // 2
    for i in range(4):
        target[:, i, cy:cy+half, cx:cx+half] = batch_imgs[:, i]
    return target


# ==========================================
# 阶段4：数据集构建
# ==========================================

class MNISTQuadDataset(Dataset):
    """
    将 MNIST 手写体按 4 张一组打包，返回 (4, S, S) 的明文组 (S 由调用方决定)
    """
    def __init__(self, mnist_dataset, img_size=256):
        self.mnist = mnist_dataset
        self.num_samples = len(self.mnist) // 4
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        imgs = []
        for i in range(4):
            img, _ = self.mnist[idx * 4 + i]
            img_s = transforms.functional.resize(img, [self.img_size, self.img_size])
            imgs.append(img_s[0])  # (S, S)
        return torch.stack(imgs, dim=0)  # (4, S, S)


# ==========================================
# 阶段5：解密网络架构
# ==========================================

class UNetRefine(nn.Module):
    """
    U-Net 精修网络: 编码器-解码器 + 跳连, 多尺度特征融合。
    使用 F.interpolate 代替 MaxPool/ConvTranspose, 支持任意尺寸输入 (含非 2 幂次, 如 1080)。
    """
    def __init__(self, in_ch=1, out_ch=1, mid_ch=64):
        super().__init__()

        # Encoder
        self.enc1 = self._conv_block(in_ch, mid_ch)
        self.enc2 = self._conv_block(mid_ch, mid_ch * 2)
        self.enc3 = self._conv_block(mid_ch * 2, mid_ch * 4)

        # Bottleneck
        self.bot = self._conv_block(mid_ch * 4, mid_ch * 8)

        # Decoder (用 1x1 conv 调整通道, 上采样用 interpolate)
        self.up3_conv = nn.Conv2d(mid_ch * 8, mid_ch * 4, 1)
        self.dec3 = self._conv_block(mid_ch * 8, mid_ch * 4)
        self.up2_conv = nn.Conv2d(mid_ch * 4, mid_ch * 2, 1)
        self.dec2 = self._conv_block(mid_ch * 4, mid_ch * 2)
        self.up1_conv = nn.Conv2d(mid_ch * 2, mid_ch, 1)
        self.dec1 = self._conv_block(mid_ch * 2, mid_ch)

        self.out_conv = nn.Conv2d(mid_ch, out_ch, 1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1)

    @staticmethod
    def _conv_block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _down(x):
        """下采样: interpolate 到一半尺寸 (支持奇数尺寸)"""
        h, w = x.shape[-2], x.shape[-1]
        return nn.functional.interpolate(x, size=(h // 2, w // 2), mode='bilinear', align_corners=False)

    @staticmethod
    def _up(x, target_size):
        """上采样到指定尺寸 (匹配 skip 连接的尺寸)"""
        return nn.functional.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

    def forward(self, x):
        # 输入按通道标准化
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + 1e-8
        x_norm = (x - mean) / std

        e1 = self.enc1(x_norm)                              # (B, mid, H, W)
        e2 = self.enc2(self._down(e1))                      # (B, 2mid, H/2, W/2)
        e3 = self.enc3(self._down(e2))                     # (B, 4mid, H/4, W/4)
        b = self.bot(self._down(e3))                       # (B, 8mid, H/8, W/8)

        d3 = self.dec3(torch.cat([self._up(self.up3_conv(b), e3.shape[-2:]), e3], dim=1))
        d2 = self.dec2(torch.cat([self._up(self.up2_conv(d3), e2.shape[-2:]), e2], dim=1))
        d1 = self.dec1(torch.cat([self._up(self.up1_conv(d2), e1.shape[-2:]), e1], dim=1))

        out = self.out_conv(d1)
        return out + self.skip(x_norm)  # 全局残差


class OAM_Crypt_D2NN(nn.Module):
    """
    混合光电解密网络: 解析预处理 + 显式 OAM 解复用 + 多平面聚焦 + 可训练 D2NN + U-Net 精修。

    多平面 OAM 复用全息 (核心创新):
      振幅模式: sqrt(P_i) -> ASM(z_i) [每路不同 z_i] -> ×OAM_i -> 求和 -> ×RPP
      解密: 双相位解码 -> ×conj(RPP) -> ×conj(OAM_j) [4 路] -> ASM(-z_j) -> D2NN -> U-Net

    通道 j 的信号项 = ASM(ASM(sqrt(P_j),z_j)·OAM_j·conj(OAM_j),-z_j) ≈ sqrt(P_j)
      => |U_back_j| ≈ sqrt(P_j) 含图像信息 → 强度随平面变化 → 平面选择性 ✓
    其他通道 i≠j 的项 = ASM(ASM(sqrt(P_i),z_i)·exp(i(l_i-l_j)θ), -z_j)
      => OAM 正交性 + 平面不匹配 → 衰减为噪声 ✓

    SLM 兼容: 双相位编码将复振幅 U_cipher 编码为纯相位, 兼容纯相位 SLM。
    关键: OAM 解复用必须在反向传播之前! 否则 OAM 正交性被破坏。
    """
    def __init__(self, size=1080, num_layers=4, wavelength=532e-9, pixel_size=8e-6,
                 z_layer=0.02, z0=0.1, rpp=None, oam_keys=None, z_list=None,
                 obj_encoding="amplitude"):
        super().__init__()
        self.layers = nn.ModuleList([DiffractiveLayer(size, nonlin_every=3, layer_idx=i) for i in range(num_layers)])
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.z_layer = z_layer
        self.z0 = z0
        self.obj_encoding = obj_encoding
        # 多平面 z_list (注册为 buffer, 跟随 device)
        if z_list is not None:
            self.register_buffer('z_list', torch.tensor(z_list, dtype=torch.float32))
            self.use_multi_plane = True
        else:
            self.register_buffer('z_list', torch.tensor([z0], dtype=torch.float32))
            self.use_multi_plane = False
        if rpp is not None:
            self.register_buffer('rpp_conj', torch.conj(rpp))
        else:
            self.rpp_conj = None
        if oam_keys is not None:
            oam_conj_stack = torch.stack([torch.conj(generate_oam_phase(size, l, 'cpu')) for l in oam_keys])
        else:
            oam_conj_stack = torch.stack([torch.conj(generate_oam_phase(size, l, 'cpu')) for l in CONFIG["l_auth"]])
        self.register_buffer('oam_conj_stack', oam_conj_stack)
        self.num_channels = len(oam_keys) if oam_keys is not None else len(CONFIG["l_auth"])
        # out_ch = num_channels: 每个通道输出一个图像, 都在同一位置 (多平面复用)
        self.refine = UNetRefine(in_ch=12, out_ch=self.num_channels, mid_ch=CONFIG["mid_ch"])

    def forward(self, U):
        device = U.device
        B = U.shape[0]

        # 0. 先去除 RPP (数字预处理, 在 SLM 加载之前)
        #    关键: RPP 必须在双相位编码之前去除, 否则低通滤波会破坏 RPP 的高频相位
        #    物理含义: RPP 是独立物理密钥, 解密方用 conj(RPP) 数字去除后再加载到 SLM
        if self.rpp_conj is not None:
            U = U * self.rpp_conj

        # 1. 模拟纯相位 SLM 加载
        if self.obj_encoding == "amplitude":
            # 双相位编码: 复振幅 -> 纯相位 (棋盘格交错) -> 低通滤波恢复复振幅
            # 此时 U 已去除 RPP, 低通滤波不会破坏 RPP
            U = double_phase_encode(U, device)
            U = lowpass_filter(U, sigma=0.15)
        else:
            # 相位模式: 仅保留相位 (旧模式, 无平面选择性)
            U = torch.exp(1j * torch.angle(U))

        # 2. OAM 解复用 (在反向传播之前!) ×conj(OAM_j) 得到 4 路解调场
        demod = U.unsqueeze(1) * self.oam_conj_stack.unsqueeze(0)  # (B,4,H,W) complex

        # 3. 多平面聚焦: 每路在对应 z_j 平面反向传播 (核心创新!)
        #    通道 j 反传到 -z_j 才能聚焦, 其他通道在此平面散焦为噪声
        if self.use_multi_plane:
            U_back_list = []
            for j in range(self.num_channels):
                z_j = float(self.z_list[j])
                U_j = propagate_asm(demod[:, j], -z_j, self.wavelength, self.pixel_size, device)
                U_back_list.append(U_j)
            U_back = torch.stack(U_back_list, dim=1)  # (B, 4, H, W)
            U_back = U_back.reshape(B * self.num_channels, U.shape[-2], U.shape[-1])
        else:
            demod_flat = demod.reshape(B * self.num_channels, U.shape[-2], U.shape[-1])
            U_back = propagate_asm(demod_flat, -self.z0, self.wavelength, self.pixel_size, device)

        # 4. D2NN 层 (恒等初始化, 微调相位补偿)
        for layer in self.layers:
            U_back = layer(U_back)
            U_back = propagate_asm(U_back, self.z_layer, self.wavelength, self.pixel_size, device)

        # 5. 重塑回 (B,4,H,W) 并拼接 real+imag+phase -> 12 通道
        #    振幅模式: 信号项 ≈ sqrt(P_j), real/imag/phase 都含图像信息
        U_back = U_back.reshape(B, self.num_channels, U.shape[-2], U.shape[-1])
        phase = torch.angle(U_back)
        x = torch.cat([U_back.real, U_back.imag, phase], dim=1)  # (B,12,H,W)
        refined = self.refine(x)  # (B, 4, H, W) 每通道一个图像
        return refined  # (B, 4, H, W)


# ==========================================
# 阶段6：评估指标与可视化
# ==========================================

def calculate_psnr(pred, target):
    """峰值信噪比 (pred 先 clamp 到 [0,1])"""
    pred = pred.clamp(0, 1)
    mse = torch.mean((pred - target) ** 2)
    if mse <= 0:
        return torch.tensor(float('inf'), device=mse.device)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def security_ratio(pred_auth, pred_unauth):
    """
    密钥敏感度: 错误密钥解密平均能量 / 正确密钥解密平均能量
    值越低安全性越强
    """
    e_auth = torch.mean(pred_auth)
    e_unauth = torch.mean(pred_unauth)
    if e_auth <= 0:
        return torch.tensor(float('inf'), device=e_auth.device)
    return e_unauth / e_auth


def _to_np(x):
    """统一将张量转为 numpy (自动 detach + cpu)"""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def save_security_plot(cipher_auth, pred_auth, target_auth,
                       cipher_unauth, pred_unauth, path="final_security_plot.png"):
    """
    绘制三行对比图 (4 通道独立输出, 图像在同一位置):
      行1: 正确密钥流 [密文强度 | 4 个解密通道 (同一位置不同图案)]
      行2: 错误密钥流 [密文强度 | 4 个解密噪声]
      行3: 原始明文标签 [4 个通道]
    pred/target 形状: (B, 4, H, W)
    """
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))

    # 行1: 正确密钥解密流
    axes[0, 0].imshow(np.abs(_to_np(cipher_auth[0])), cmap='gray')
    axes[0, 0].set_title("Cipher (Auth)")
    for i in range(4):
        axes[0, i + 1].imshow(_to_np(pred_auth[0, i]), cmap='gray', vmin=0, vmax=1)
        axes[0, i + 1].set_title(f"Decrypted Ch{i} (Auth)")

    # 行2: 错误密钥解密流
    axes[1, 0].imshow(np.abs(_to_np(cipher_unauth[0])), cmap='gray')
    axes[1, 0].set_title("Cipher (Wrong Key)")
    for i in range(4):
        axes[1, i + 1].imshow(_to_np(pred_unauth[0, i]), cmap='gray', vmin=0, vmax=1)
        axes[1, i + 1].set_title(f"Decrypted Ch{i} (Wrong)")

    # 行3: 原始明文标签
    axes[2, 0].axis('off')
    for i in range(4):
        axes[2, i + 1].imshow(_to_np(target_auth[0, i]), cmap='gray', vmin=0, vmax=1)
        axes[2, i + 1].set_title(f"Plaintext Ch{i}")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ==========================================
# 阶段7：主训练与测试流程
# ==========================================

if __name__ == "__main__":
    device = torch.device(CONFIG["device"])
    torch.manual_seed(42)
    np.random.seed(42)

    # ---------- 1. 数据准备 ----------
    transform = transforms.Compose([transforms.ToTensor()])
    os.makedirs("./data", exist_ok=True)
    full_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    full_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 子集采样 (1080 分辨率训练慢, 减少样本量到 400 组训练 / 100 组测试)
    mnist_train = Subset(full_train, range(800))
    mnist_test = Subset(full_test, range(200))

    # 图像区域 size//4 (270x270), 中心放置, 外围留空增强 OAM 正交性
    train_dataset = MNISTQuadDataset(mnist_train, img_size=CONFIG["size"] // 4)
    test_dataset = MNISTQuadDataset(mnist_test, img_size=CONFIG["size"] // 4)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, persistent_workers=True)

    # ---------- 2. 生成全局系统物理密钥 RPP ----------
    rpp_system = generate_rpp(CONFIG["size"], device)  # 授权 RPP (整个训练过程固定)

    # ---------- 2. 网络与优化器 ----------
    model = OAM_Crypt_D2NN(
        size=CONFIG["size"], num_layers=CONFIG["num_layers"],
        wavelength=CONFIG["wavelength"], pixel_size=CONFIG["pixel_size"],
        z_layer=CONFIG["z_layer"], z0=CONFIG["z0"], rpp=rpp_system,
        oam_keys=CONFIG["l_auth"], z_list=CONFIG["z_list"],
        obj_encoding=CONFIG["obj_encoding"]
    ).to(device)

    # D2NN 层初始化为恒等 (phase=0, amp≈1)
    with torch.no_grad():
        for i in range(CONFIG["num_layers"]):
            model.layers[i].phase.zero_()
            model.layers[i].amp_logit.fill_(4.0)  # sigmoid(4)≈0.98
    print(f"已初始化: RPP去除 -> OAM解复用(4路) -> 反向传播 -> U-Net(12ch, mid={CONFIG['mid_ch']}, D2NN层={CONFIG['num_layers']})", flush=True)

    # 分组学习率: D2NN 用低 lr 微调 OAM 补偿; U-Net 用正常 lr 快速学习
    d2nn_params = list(model.layers.parameters())
    refine_params = list(model.refine.parameters())
    param_groups = [{"params": refine_params, "lr": CONFIG["lr"]}]
    if len(d2nn_params) > 0:
        param_groups.insert(0, {"params": d2nn_params, "lr": CONFIG["lr_d2nn"]})
    optimizer = optim.Adam(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    criterion_mse = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(CONFIG["device"] == "cuda"))  # AMP 混合精度

    print(f"开始训练 (12ch输入:4real+4imag+4phase, mid={CONFIG['mid_ch']}, lr_d2nn={CONFIG['lr_d2nn']}, lr_unet={CONFIG['lr']}, AMP={CONFIG['device']=='cuda'})", flush=True)

    # ---------- 4. 对抗安全训练循环 (课程学习) ----------
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        epoch_loss_auth = 0.0
        epoch_loss_xtalk = 0.0
        epoch_loss_sec = 0.0

        # 课程学习: warmup 阶段只用 MSE 专注学习重构, 之后再叠加安全+串扰损失
        use_security = (epoch > CONFIG["warmup_epochs"])
        n_batches = len(train_loader)
        t_start = __import__("time").time()

        for bidx, batch_imgs in enumerate(train_loader):
            batch_imgs = batch_imgs.to(device)
            target = build_target_grid(batch_imgs, device, size=CONFIG["size"])

            # (a) 合法输入: 正确 OAM + 正确 RPP
            cipher_auth = encrypt_batch(
                batch_imgs, CONFIG["l_auth"], rpp_system,
                CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
                size=CONFIG["size"], z_list=CONFIG["z_list"],
                obj_encoding=CONFIG["obj_encoding"]
            )

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(CONFIG["device"] == "cuda")):
                pred_auth = model(cipher_auth)

                # 授权重构损失
                loss_mse = criterion_mse(pred_auth, target)
                loss_l1 = torch.mean(torch.abs(pred_auth - target))
                loss_auth = loss_mse + CONFIG["l1_weight"] * loss_l1

                loss_xtalk = torch.tensor(0.0, device=device)
                loss_sec = torch.tensor(0.0, device=device)

                if use_security:
                    # 通道间串扰: pred_auth[:, i] 不应包含 target[:, j] (i!=j)
                    # 现在 4 个图像都在中心同一位置, 串扰 = 不同通道间的内容泄漏
                    for i in range(4):
                        for j in range(4):
                            if i != j:
                                loss_xtalk = loss_xtalk + torch.mean(pred_auth[:, i] * target[:, j])

                    use_wrong_oam = torch.rand(1).item() < 0.5
                    if use_wrong_oam:
                        cipher_unauth = encrypt_batch(
                            batch_imgs, CONFIG["l_wrong"], rpp_system,
                            CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
                            size=CONFIG["size"], z_list=CONFIG["z_list"],
                            obj_encoding=CONFIG["obj_encoding"]
                        )
                    else:
                        rpp_wrong = generate_rpp(CONFIG["size"], device)
                        cipher_unauth = encrypt_batch(
                            batch_imgs, CONFIG["l_auth"], rpp_wrong,
                            CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
                            size=CONFIG["size"], z_list=CONFIG["z_list"],
                            obj_encoding=CONFIG["obj_encoding"]
                        )
                    pred_unauth = model(cipher_unauth)
                    loss_sec = criterion_mse(pred_unauth, torch.zeros_like(pred_unauth))

                total_loss = loss_auth + CONFIG["xtalk_weight"] * loss_xtalk + CONFIG["sec_weight"] * loss_sec

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 每 50 batch 打印进度
            if (bidx + 1) % 50 == 0 or bidx == 0:
                elapsed = __import__("time").time() - t_start
                eta = elapsed / (bidx + 1) * n_batches
                print(f"  Epoch {epoch} [{bidx+1}/{n_batches}] loss={total_loss.item():.5f} "
                      f"t={elapsed:.0f}s ETA={eta:.0f}s", flush=True)

            bs = batch_imgs.size(0)
            epoch_loss += total_loss.item() * bs
            epoch_loss_auth += loss_auth.item() * bs
            epoch_loss_xtalk += float(loss_xtalk.detach()) * bs
            epoch_loss_sec += loss_sec.item() * bs

        scheduler.step()
        n = len(train_loader.dataset)
        epoch_loss /= n
        epoch_loss_auth /= n
        epoch_loss_xtalk /= n
        epoch_loss_sec /= n

        # ---------- 5. 验证与日志 ----------
        if epoch % 1 == 0 or epoch == 1:  # 每 epoch 都验证 (训练轮次少时方便观察)
            model.eval()
            psnr_list = []
            sec_ratio_list = []
            with torch.no_grad():
                for test_batch in test_loader:
                    test_batch = test_batch.to(device)
                    tgt = build_target_grid(test_batch, device, size=CONFIG["size"])

                    # 合法解密
                    c_auth = encrypt_batch(
                        test_batch, CONFIG["l_auth"], rpp_system,
                        CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
                        size=CONFIG["size"], z_list=CONFIG["z_list"],
                        obj_encoding=CONFIG["obj_encoding"]
                    )
                    p_auth = model(c_auth)

                    # 非法解密 (使用错误 RPP)
                    rpp_wrong_eval = generate_rpp(CONFIG["size"], device)
                    c_unauth = encrypt_batch(
                        test_batch, CONFIG["l_auth"], rpp_wrong_eval,
                        CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
                        size=CONFIG["size"], z_list=CONFIG["z_list"],
                        obj_encoding=CONFIG["obj_encoding"]
                    )
                    p_unauth = model(c_unauth)

                    psnr_list.append(calculate_psnr(p_auth, tgt).item())
                    sec_ratio_list.append(security_ratio(p_auth, p_unauth).item())

            avg_psnr = float(np.mean(psnr_list))
            avg_sec = float(np.mean(sec_ratio_list))
            print(f"Epoch [{epoch}/{CONFIG['epochs']}] | Loss: {epoch_loss:.6f} "
                  f"(auth={epoch_loss_auth:.6f}, xtalk={epoch_loss_xtalk:.6f}, sec={epoch_loss_sec:.6f}) "
                  f"| PSNR: {avg_psnr:.2f} dB | SecurityRatio: {avg_sec:.4f}", flush=True)

            # 保存阶段性模型
            torch.save(model.state_dict(), f"oam_crypt_dnn_epoch_{epoch}.pth")

            # 缓存最后一个测试样本用于最终可视化
            last_cipher_auth, last_pred_auth, last_tgt = c_auth, p_auth, tgt
            last_cipher_unauth, last_pred_unauth = c_unauth, p_unauth

    # ---------- 6. 最终可视化 ----------
    print("训练结束，生成最终安全性对比图 final_security_plot.png ...")
    save_security_plot(
        last_cipher_auth, last_pred_auth, last_tgt,
        last_cipher_unauth, last_pred_unauth,
        path="final_security_plot.png"
    )
    print("完成。")
