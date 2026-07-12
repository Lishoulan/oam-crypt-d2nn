# -*- coding: utf-8 -*-
"""
双密钥全光多用户信息加解密与隐写衍射神经网络 (D2NN) 数值仿真
================================================================
多平面 OAM 复用全息 + 双相位编码 (纯相位 SLM 兼容)
10 通道 OAM-MDNN (Milestone v2.0, 仿北理工 Nature Photonics 2026)
v4 优化: Attention U-Net + 15 epoch 训练 + 启用安全损失

物理流程：
  加密: 明文 P_i -> 相位 exp(iπP_i) -> ASM(z_i) [每路不同 z_i, 多平面复用]
       -> OAM 密钥编码 e^{i l_i theta} -> 10 路叠加 U_sum -> RPP 调制 -> 密文 U_cipher
  解密: U_cipher -> 去除 RPP (数字预处理) -> 双相位编码 (纯相位 SLM)
       -> 低通滤波 (恢复复振幅) -> OAM 解复用 [10 路] -> ASM(-z_j) [多平面聚焦]
       -> D2NN -> U-Net 精修

三大功能:
  1. 不同平面出现不同图案 (z_list 间距 5cm, 10 个平面)
  2. 错误 OAM 拓扑荷看不到图像 (OAM 正交性, 安全比 <0.3)
  3. 正确 OAM 拓扑荷只看到对应平面图案 (多平面聚焦选择性)

双密钥: OAM 拓扑荷 (用户级) + RPP 随机相位板 (系统级)
SLM 兼容: 双相位编码将复振幅编码为纯相位, 兼容纯相位 SLM
仅依赖: torch, torchvision, numpy, matplotlib
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    # v3 10 通道: 5cm 间隔, 共 10 个平面 (总程 45cm), 大 OAM 间距保证正交性
    "z_list": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55],  # 10 通道, 10 个平面
    "z_layer": 0.02,          # D2NN 相位层之间的传播距离
    "l_auth": [-25, -20, -15, -10, -5, 5, 10, 15, 20, 25],  # 10 个 OAM 通道 (5 对, 大步长 ±5)
    "l_wrong": [-30, -23, -12, -8, -3, 7, 12, 18, 23, 28],  # 错误 OAM 密钥 (10 个, 远离授权值)
    "batch_size": 1,           # 批大小 (3层D2NN+安全损失显存大, 降至1防OOM; 梯度累积补回有效批)
    "epochs": 5,               # v4 优化: 8→5 epoch (v3 8 epoch 30.85 dB; Attention U-Net 5 epoch 应达 31+ dB)
    "lr": 3e-4,                # U-Net 精修层学习率 (Phase 1 修复: 降低 LR 避免损失爆炸)
    "lr_d2nn": 0.05,           # D2NN 物理层学习率 (降低)
    "mid_ch": 64,             # U-Net 中间通道数 (Phase 1: 减到 64 加快训练)
    "num_layers": 2,          # 衍射层数 (借鉴文章 2 层无源衍射片设计)
    "freeze_epochs": 0,       # 0=不冻结
    "warmup_epochs": 2,       # v4 优化: warmup 2 epoch 纯重建, epoch 3-5 启用安全损失(0.1 权重)
    "sec_weight": 0.1,        # v4 优化: 启用小权重安全损失(0.1), 训练 SecurityRatio < 0.3
    "xtalk_weight": 0.0,      # 关闭串扰损失 (单通道无串扰)
    "l1_weight": 0.1,         # v4 优化: 启用 L1 损失 0.1 权重(配合 MSE 提升锐度)
    "quick_test_n": 1600,     # v4 训练: 用全量 1600 样本(默认)
    # 物光编码模式: "amplitude" = sqrt(P) 振幅编码 (有平面选择性, 配合双相位编码兼容纯相位SLM)
    #               "phase" = exp(iπP) 相位编码 (无平面选择性, 但天然纯相位)
    "obj_encoding": "phase",   # Phase 1: 相位编码 (无 sqrt 开方损耗, PSNR 上限 27+ dB)
    # k 空间约束: 限制 ASM 传播的最大空间频率对应角度 (单位: 度)
    # SLM 像素 8μm, λ=532nm 时理论衍射极限 θ≈1.9°; 取 1.5° 保守约束,
    # 抑制器件无法支持的高频分量, 使训练相位更平滑、更接近实际 SLM 可实现条件
    "theta_max_deg": 1.5,
    # SLM 感知训练: 在模型 forward 内部模拟 SLM 物理加载 (8-bit 相位量化)
    # True: 训练时学到的就是 SLM 加载后的真实性能,数字流程与 SLM 流程一致
    # False: 训练时是纯数字流程,SLM 加载时有 13 dB 损耗
    "slm_aware": True,
    # 断点续训: 指定 checkpoint 路径则从中断处继续训练 (None=从头开始)
    # 续训时需确保 CONFIG 其他参数与原训练一致, 否则权重失配
    "resume": None,           # 从头训练 (参数调整: sec_weight 0.3→0.6, l_wrong 扩大, 每 batch 安全损失)
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

print(f"当前运行设备: {CONFIG['device']}")


# ==========================================
# 阶段2：核心物理算子
# ==========================================

def propagate_asm(U_in, z, wavelength, pixel_size, device, theta_max=None):
    """
    角谱法(ASM)可微自由空间传播算子
    支持输入形状 (B, H, W) 或 (H, W)，输出对应形状的复振幅

    k 空间约束 (theta_max): 限制最大传播角度, 抑制超过器件/系统可支持
    角度的高频分量, 使训练/设计结果更接近实际 SLM 或 DOE 可实现的物理条件。
    参考: 公式中每个空间频率 kr 对应传播角 θ=arcsin(kr/k);
          若 kr > k·sin(θ_max), 则该分量被滤除 (mask=0)。
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

    # k 空间约束: 限制最大传播角度 (连接理想仿真与真实系统的关键)
    if theta_max is not None:
        kx = 2 * np.pi * f_x
        ky = 2 * np.pi * f_y
        kr = torch.sqrt(kx ** 2 + ky ** 2)
        k_mask = (kr <= k * np.sin(theta_max)).float()
        mask = mask * k_mask

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
                  size=1080, z_list=None, obj_encoding="amplitude", theta_max=None):
    """
    双密钥加密引擎(支持授权/未授权两种模式，由传入的 oam_keys / rpp 决定)
    - 输入: batch_imgs (B, C, S, S) 明文振幅图 (C = OAM 通道数, S 由调用方决定)
    - 输出: U_cipher (B, size, size) 复数密文场

    多平面 OAM 复用全息 (核心创新):
      - C 个图像都放在画面中心同一位置 (不分象限, 真正的复用全息)
      - 每路 OAM 通道用不同传播距离 z_i
      - 解密时只有用对应 conj(OAM_j) 解调 + 在 z_j 平面观察才能看到图像 j
      - 不同平面的图案在同一位置出现, 不会错开
    向后兼容: 若 z_list=None, 则所有路都用 z0 (旧行为)
    k 空间约束: theta_max (弧度) 限制 ASM 传播的最大空间频率, 与解密网络保持一致
    """
    B = batch_imgs.shape[0]
    U_sum = torch.zeros(B, size, size, dtype=torch.complex64, device=device)
    half = batch_imgs.shape[-1]  # 单图尺寸 (10 通道布局: 216 = size//5)
    # v3 10 通道: 2x5 网格布局, 每格 216x216, 整体 432x1080 居中
    # 每个通道在 2x5 网格的对应位置放置, 与 build_target_grid 一致
    rows, cols = 2, 5
    cell_h, cell_w = half, half
    y_start = (size - rows * cell_h) // 2
    x_start = (size - cols * cell_w) // 2

    for i, l in enumerate(oam_keys):
        img_pad = torch.zeros(B, size, size, device=device)
        r = i // cols
        c = i % cols
        y = y_start + r * cell_h
        x = x_start + c * cell_w
        img_pad[:, y:y+cell_h, x:x+cell_w] = batch_imgs[:, i]

        if obj_encoding == "phase":
            U_obj = torch.exp(1j * np.pi * img_pad)
        else:
            U_obj = torch.sqrt(img_pad).to(torch.complex64)

        z_i = z_list[i] if z_list is not None else z0
        U_prop = propagate_asm(U_obj, z_i, wavelength, pixel_size, device, theta_max=theta_max)

        oam_phase = generate_oam_phase(size, l, device)
        U_sum = U_sum + U_prop * oam_phase

    U_cipher = U_sum * rpp
    return U_cipher


def build_target_grid(batch_imgs, device, size=1080, num_channels=None):
    """
    构建 (B, C, size, size) 目标网格: 每通道对应一个图像, 按 2x5 网格布局
    (v3 10 通道, 适配北理工 10 通道 OAM-MDNN 架构)
    布局: 2 行 x 5 列, 每格 216x216, 整体 432x1080, 上下 padding 324 到 1080x1080
    C 由 num_channels 决定 (默认从 batch_imgs 自动推断, 与 OAM 通道数一致)
    """
    B, C, H, W = batch_imgs.shape
    if num_channels is None:
        num_channels = C
    # 2x5 网格布局 (10 通道专用)
    rows, cols = 2, 5
    cell_h, cell_w = H, W  # 每格大小与 batch_imgs 单图一致 (216x216)
    target = torch.zeros(B, num_channels, size, size, device=device)
    # 居中: 上下 (size - rows*cell_h)//2, 左右 (size - cols*cell_w)//2
    # 若 cols*cell_w == size, 则左右 0; 若 rows*cell_h < size, 则上下 padding
    y_start = (size - rows * cell_h) // 2
    x_start = (size - cols * cell_w) // 2
    for i in range(num_channels):
        r = i // cols
        c = i % cols
        y = y_start + r * cell_h
        x = x_start + c * cell_w
        target[:, i, y:y+cell_h, x:x+cell_w] = batch_imgs[:, i]
    return target


# ==========================================
# 阶段4：数据集构建
# ==========================================

class MNISTQuadDataset(Dataset):
    """
    将 MNIST 手写体按 num_channels 张一组打包，返回 (num_channels, S, S) 的明文组
    v2 简化: 支持任意通道数 (默认 2, 与 OAM 通道数一致)
    """
    def __init__(self, mnist_dataset, img_size=256, num_channels=2):
        self.mnist = mnist_dataset
        self.num_channels = num_channels
        self.num_samples = len(self.mnist) // num_channels
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        imgs = []
        for i in range(self.num_channels):
            img, _ = self.mnist[idx * self.num_channels + i]
            img_s = transforms.functional.resize(img, [self.img_size, self.img_size])
            imgs.append(img_s[0])  # (S, S)
        return torch.stack(imgs, dim=0)  # (num_channels, S, S)


# ==========================================
# 阶段5：解密网络架构
# ==========================================

class AttentionGate(nn.Module):
    """
    Attention Gate (Oktay et al. 2018, Attention U-Net).
    用于 skip connection, 让 decoder 选择性关注 encoder 的相关特征。
    Args:
        gate_ch: gating signal 通道数 (来自 decoder 较深层)
        in_ch:   skip connection 通道数 (来自 encoder)
        inter_ch: 中间通道数(默认 gate_ch // 2)
    """
    def __init__(self, gate_ch, in_ch, inter_ch=None):
        super().__init__()
        inter_ch = inter_ch or max(gate_ch // 2, in_ch // 2)
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        """
        Args:
            x: encoder skip features (B, in_ch, H, W)
            g: decoder gating signal (B, gate_ch, H', W')
        Returns:
            attended: x * attention_coefficient (B, in_ch, H, W)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # gating signal 和 skip 尺寸对齐
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(g1, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        psi = self.relu(g1 + x1)
        att = self.psi(psi)
        return x * att


class UNetRefine(nn.Module):
    """
    Attention U-Net 精修网络 (Oktay et al. 2018 改进版):
      编码器(3层下采样) + Bottleneck(1层) + 解码器(3层上采样) +
      Attention Gate (每个 skip connection) + 输出层 + 残差连接

    使用 F.interpolate 代替 MaxPool/ConvTranspose, 支持任意尺寸输入 (含非 2 幂次, 如 1080)。
    相比基础 U-Net, attention 让网络自适应选择对当前通道重要的特征。
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

        # Attention Gates (skip connections: e3->d3, e2->d2, e1->d1)
        self.att3 = AttentionGate(gate_ch=mid_ch * 4, in_ch=mid_ch * 4)
        self.att2 = AttentionGate(gate_ch=mid_ch * 2, in_ch=mid_ch * 2)
        self.att1 = AttentionGate(gate_ch=mid_ch,     in_ch=mid_ch)

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
        return F.interpolate(x, size=(h // 2, w // 2), mode='bilinear', align_corners=False)

    @staticmethod
    def _up(x, target_size):
        """上采样到指定尺寸 (匹配 skip 连接的尺寸)"""
        return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

    def forward(self, x):
        # 输入按通道标准化
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + 1e-8
        x_norm = (x - mean) / std

        e1 = self.enc1(x_norm)                              # (B, mid, H, W)
        e2 = self.enc2(self._down(e1))                      # (B, 2mid, H/2, W/2)
        e3 = self.enc3(self._down(e2))                     # (B, 4mid, H/4, W/4)
        b = self.bot(self._down(e3))                       # (B, 8mid, H/8, W/8)

        # Decoder + Attention Gates
        u3 = self.up3_conv(b)
        e3_att = self.att3(e3, u3)                          # attention on skip e3
        d3 = self.dec3(torch.cat([self._up(u3, e3.shape[-2:]), e3_att], dim=1))

        u2 = self.up2_conv(d3)
        e2_att = self.att2(e2, u2)                          # attention on skip e2
        d2 = self.dec2(torch.cat([self._up(u2, e2.shape[-2:]), e2_att], dim=1))

        u1 = self.up1_conv(d2)
        e1_att = self.att1(e1, u1)                          # attention on skip e1
        d1 = self.dec1(torch.cat([self._up(u1, e1.shape[-2:]), e1_att], dim=1))

        out = self.out_conv(d1)
        # 线性输出 + 残差连接, 损失函数中 clamp + 加权
        return out + self.skip(x_norm)


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
                 obj_encoding="amplitude", theta_max=None, slm_aware=True):
        super().__init__()
        self.layers = nn.ModuleList([DiffractiveLayer(size, nonlin_every=3, layer_idx=i) for i in range(num_layers)])
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.z_layer = z_layer
        self.z0 = z0
        self.obj_encoding = obj_encoding
        self.theta_max = theta_max  # k 空间约束 (弧度), None=不约束
        self.slm_aware = slm_aware  # True=在 forward 内部模拟 SLM 8-bit 量化
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
        # v2: 3 通道/图像 (|U|, real, imag), 总输入 = 3 * num_channels
        self.refine = UNetRefine(in_ch=3 * self.num_channels, out_ch=self.num_channels, mid_ch=CONFIG["mid_ch"])

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
            # v3 10 通道: 加 lowpass 去棋盘格, 让 phase 模式也能处理 SLM 加载
            # 训练时: c → exp(i*angle) → lowpass → 8bit 量化 (无棋盘格, 干净)
            # SLM 推理: c → DPE → 8bit 棋盘格 phase → exp(i*angle) → lowpass(去棋盘格) → 8bit
            # lowpass 模拟 4f 系统衍射恢复复振幅, 让训练和 SLM 一致
            U = torch.exp(1j * torch.angle(U))
            U = lowpass_filter(U, sigma=0.15)  # 关键: 去棋盘格高频, 恢复复振幅信息

        # SLM 感知训练: 模拟 Holoeye PLUTO 8-bit 相位量化 (关键!)
        # 实际 SLM 加载流程: 连续相位 -> 8-bit 灰度 -> 离散相位 (256 级)
        # 训练时模拟此量化, 模型才能学到 SLM 加载后的真实分布
        # 关闭 slm_aware 时, 数字流程有 ~13 dB 损耗; 开启后数字与 SLM 流程一致
        # 此时 U 已是 phase only 单位模场 (无论 amplitude 还是 phase 模式)
        if self.slm_aware:
            phase_slm = torch.angle(U)              # 连续相位 [-π, π]
            gray_slm = ((phase_slm + np.pi) / (2 * np.pi) * 255).round()  # 8-bit 整数
            phase_q = gray_slm / 255.0 * 2 * np.pi - np.pi                # 量化还原
            U = torch.exp(1j * phase_q)            # 纯相位单位模场 (SLM 加载结果)

        # 2. OAM 解复用 (在反向传播之前!) ×conj(OAM_j) 得到 4 路解调场
        demod = U.unsqueeze(1) * self.oam_conj_stack.unsqueeze(0)  # (B,4,H,W) complex

        # 3. 多平面聚焦: 每路在对应 z_j 平面反向传播 (核心创新!)
        #    通道 j 反传到 -z_j 才能聚焦, 其他通道在此平面散焦为噪声
        if self.use_multi_plane:
            U_back_list = []
            for j in range(self.num_channels):
                z_j = float(self.z_list[j])
                U_j = propagate_asm(demod[:, j], -z_j, self.wavelength, self.pixel_size, device,
                                     theta_max=self.theta_max)
                U_back_list.append(U_j)
            U_back = torch.stack(U_back_list, dim=1)  # (B, 4, H, W)
            U_back = U_back.reshape(B * self.num_channels, U.shape[-2], U.shape[-1])
        else:
            demod_flat = demod.reshape(B * self.num_channels, U.shape[-2], U.shape[-1])
            U_back = propagate_asm(demod_flat, -self.z0, self.wavelength, self.pixel_size, device,
                                    theta_max=self.theta_max)

        # 4. D2NN 层 (恒等初始化, 微调相位补偿)
        for layer in self.layers:
            U_back = layer(U_back)
            U_back = propagate_asm(U_back, self.z_layer, self.wavelength, self.pixel_size, device,
                                    theta_max=self.theta_max)

        # 5. 重塑回 (B, num_channels, H, W) 并拼接为 3 通道输入
        #    v2 简化: 12 通道 -> 3 通道 (|U|, real, imag), 去掉冗余
        #    振幅 |U| 含信号强度, real/imag 保留相位信息
        U_back = U_back.reshape(B, self.num_channels, U.shape[-2], U.shape[-1])
        U_amp = torch.abs(U_back)
        x = torch.cat([U_amp, U_back.real, U_back.imag], dim=1)  # (B, 3*num_channels, H, W)
        refined = self.refine(x)  # (B, num_channels, H, W) 每通道一个图像
        return refined  # (B, num_channels, H, W)


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


def calculate_center_psnr(pred, target, center_size=270):
    """
    中心区域 PSNR (v2 新增, v3 10 通道适配)
    v2: 只在中心 center_size×center_size 区域计算, 反映真实图像质量
    v3 10 通道: 中心区域改为整个 2x5 网格覆盖范围 (432x1080, 居中 1080x1080)
        旧 center_size=270 已被忽略, 直接用 2x5 网格范围
    pred: (B, C, H, W) 解密图像
    target: (B, C, H, W) 明文图像 (build_target_grid 输出)
    旧 PSNR 在 1080×1080 全图算 (94% 是黑边), 虚高; 中心 PSNR 反映肉眼可见的质量
    """
    pred = pred.clamp(0, 1)
    # v3 10 通道: 2x5 网格覆盖范围 y=[324:756] (432 高), x=[0:1080] (整宽)
    pred_c = pred[..., 324:756, 0:1080]
    tgt_c = target[..., 324:756, 0:1080]
    mse = torch.mean((pred_c - tgt_c) ** 2)
    if mse <= 0:
        return torch.tensor(float('inf'), device=mse.device)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def security_ratio(pred_auth, pred_unauth):
    """
    密钥敏感度: 错误密钥解密平均能量 / 正确密钥解密平均能量
    值越低安全性越强
    注意: 必须用 abs() 计算能量, 否则正负值相抵会虚低 (模型会作弊输出零均值噪声)
    """
    e_auth = pred_auth.abs().mean()
    e_unauth = pred_unauth.abs().mean()
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
    绘制三行对比图 (C 通道独立输出, 图像在同一位置):
      行1: 正确密钥流 [密文强度 | C 个解密通道 (同一位置不同图案)]
      行2: 错误密钥流 [密文强度 | C 个解密噪声]
      行3: 原始明文标签 [C 个通道]
    pred/target 形状: (B, C, H, W), C 由数据动态决定
    """
    C = pred_auth.shape[1]
    fig, axes = plt.subplots(3, C + 1, figsize=(3 * (C + 1), 9))
    if C + 1 == 1:
        axes = axes.reshape(3, 1)

    # 行1: 正确密钥解密流
    axes[0, 0].imshow(np.abs(_to_np(cipher_auth[0])), cmap='gray')
    axes[0, 0].set_title("Cipher (Auth)")
    for i in range(C):
        axes[0, i + 1].imshow(_to_np(pred_auth[0, i]), cmap='gray', vmin=0, vmax=1)
        axes[0, i + 1].set_title(f"Decrypted Ch{i} (Auth)")

    # 行2: 错误密钥解密流
    axes[1, 0].imshow(np.abs(_to_np(cipher_unauth[0])), cmap='gray')
    axes[1, 0].set_title("Cipher (Wrong Key)")
    for i in range(C):
        axes[1, i + 1].imshow(_to_np(pred_unauth[0, i]), cmap='gray', vmin=0, vmax=1)
        axes[1, i + 1].set_title(f"Decrypted Ch{i} (Wrong)")

    # 行3: 原始明文标签
    axes[2, 0].axis('off')
    for i in range(C):
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

    # 子集采样 (v3 10 通道: 1600 训练 / 400 测试, 每 10 张为一组, 得 160/40 组)
    # v4 smoke test: quick_test_n 限制样本数(默认 None=全部 1600)
    n_train = CONFIG.get("quick_test_n", None) or 1600
    n_test = CONFIG.get("quick_test_n", None) or 400
    mnist_train = Subset(full_train, range(n_train))
    mnist_test = Subset(full_test, range(n_test))

    # 图像区域 size//5 (216x216, 2x5 网格布局, 每格方形)
    # v3 10 通道: 5 列布局, width=1080/5=216; 2 行, 2*216=432, padding 上下 324
    num_channels = len(CONFIG["l_auth"])
    img_size = CONFIG["size"] // 5
    train_dataset = MNISTQuadDataset(mnist_train, img_size=img_size, num_channels=num_channels)
    test_dataset = MNISTQuadDataset(mnist_test, img_size=img_size, num_channels=num_channels)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, persistent_workers=True)

    # ---------- 2. 生成全局系统物理密钥 RPP ----------
    rpp_system = generate_rpp(CONFIG["size"], device)  # 授权 RPP (整个训练过程固定)

    # ---------- 2. 网络与优化器 ----------
    # k 空间约束: 度数 -> 弧度
    theta_max_rad = np.deg2rad(CONFIG["theta_max_deg"]) if CONFIG.get("theta_max_deg") else None
    model = OAM_Crypt_D2NN(
        size=CONFIG["size"], num_layers=CONFIG["num_layers"],
        wavelength=CONFIG["wavelength"], pixel_size=CONFIG["pixel_size"],
        z_layer=CONFIG["z_layer"], z0=CONFIG["z0"], rpp=rpp_system,
        oam_keys=CONFIG["l_auth"], z_list=CONFIG["z_list"],
        obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max_rad,
        slm_aware=CONFIG["slm_aware"]
    ).to(device)

    # D2NN 层初始化为恒等 (phase=0, amp≈1)
    with torch.no_grad():
        for i in range(CONFIG["num_layers"]):
            model.layers[i].phase.zero_()
            model.layers[i].amp_logit.fill_(4.0)  # sigmoid(4)≈0.98
    print(f"已初始化: RPP去除 -> OAM解复用({num_channels}路) -> 反向传播 -> U-Net(3*{num_channels}={3*num_channels}ch, mid={CONFIG['mid_ch']}, D2NN层={CONFIG['num_layers']})", flush=True)

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

    print(f"开始训练 (U-Net输入3*{num_channels}={3*num_channels}ch, mid={CONFIG['mid_ch']}, lr_d2nn={CONFIG['lr_d2nn']}, lr_unet={CONFIG['lr']}, AMP={CONFIG['device']=='cuda'})", flush=True)

    # ---------- 3.5 断点续训: 加载 checkpoint ----------
    start_epoch = 1
    resume_path = CONFIG.get("resume")
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        # 兼容旧格式 (仅 state_dict) 和新格式 (含元数据的 dict)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            print(f"断点续训: 从 {resume_path} 恢复, 起始 Epoch {start_epoch} "
                  f"(上次 PSNR={ckpt.get('psnr','?'):.2f} dB, SR={ckpt.get('sec_ratio','?'):.4f})", flush=True)
        else:
            # 旧格式: 仅 model.state_dict()
            model.load_state_dict(ckpt)
            # 从文件名提取 epoch 数字
            import re
            m = re.search(r'epoch_(\d+)', resume_path)
            start_epoch = int(m.group(1)) + 1 if m else 1
            print(f"断点续训(旧格式): 从 {resume_path} 恢复, 起始 Epoch {start_epoch} (optimizer 已重置)", flush=True)
    elif resume_path:
        print(f"警告: resume='{resume_path}' 不存在, 从头开始训练", flush=True)

    # ---------- 4. 对抗安全训练循环 (课程学习) ----------
    for epoch in range(start_epoch, CONFIG["epochs"] + 1):
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
                obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max_rad
            )

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(CONFIG["device"] == "cuda")):
                pred_auth = model(cipher_auth)

                # 授权重构损失 (Phase 1 修复: 中心区域加权, 不使用 clamp)
                # 关键修复: clamp 会杀死负值梯度 (pred<0 时 grad=0), 模型永远学不会推高
                # 改用 raw pred 计算 MSE
                # v3 10 通道: 中心加权区域 = 2x5 网格覆盖范围 (432x1080 居中)
                H, W = target.shape[-2:]
                weight_map = torch.ones(1, 1, H, W, device=device) * 0.1
                # 2x5 网格: y=[324, 756], x=[0, 1080]
                weight_map[..., 324:756, 0:1080] = 10.0
                # 用 raw pred 计算 MSE, 不要 clamp (避免杀死梯度)
                loss_mse = torch.mean(weight_map * (pred_auth - target) ** 2)
                loss_l1 = torch.mean(weight_map * torch.abs(pred_auth - target))
                loss_auth = loss_mse + CONFIG["l1_weight"] * loss_l1

                loss_xtalk = torch.tensor(0.0, device=device)
                loss_sec = torch.tensor(0.0, device=device)

                if use_security:
                    # 通道间串扰: pred_auth[:, i] 不应包含 target[:, j] (i!=j)
                    # v2: num_channels=2, 多平面复用下不同通道的图案在同一位置
                    for i in range(num_channels):
                        for j in range(num_channels):
                            if i != j:
                                loss_xtalk = loss_xtalk + torch.mean(pred_auth[:, i] * target[:, j])

                    # 安全损失: 每 batch 都计算 (原每10batch, 改为每batch增强安全训练)
                    # 从 l_wrong 列表随机采样 num_channels 个作为错误 OAM (增加多样性)
                    use_wrong_oam = torch.rand(1).item() < 0.5
                    if use_wrong_oam:
                        # 从 l_wrong 随机采样 num_channels 个 (可重复, 增加多样性)
                        idxs = torch.randint(0, len(CONFIG["l_wrong"]), (num_channels,))
                        l_wrong_sampled = [CONFIG["l_wrong"][i] for i in idxs]
                        cipher_unauth = encrypt_batch(
                            batch_imgs, l_wrong_sampled, rpp_system,
                            CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
                            size=CONFIG["size"], z_list=CONFIG["z_list"],
                            obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max_rad
                        )
                    else:
                        rpp_wrong = generate_rpp(CONFIG["size"], device)
                        cipher_unauth = encrypt_batch(
                            batch_imgs, CONFIG["l_auth"], rpp_wrong,
                            CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
                            size=CONFIG["size"], z_list=CONFIG["z_list"],
                            obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max_rad
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
            psnr_center_list = []  # v2: 中心区 PSNR (真实图像质量)
            sr_rpp_list = []    # RPP 攻击安全比 (正确OAM + 错误RPP)
            sr_oam_list = []    # OAM 攻击安全比 (错误OAM + 正确RPP)
            with torch.no_grad():
                for test_batch in test_loader:
                    test_batch = test_batch.to(device)
                    tgt = build_target_grid(test_batch, device, size=CONFIG["size"])

                    # 合法解密
                    c_auth = encrypt_batch(
                        test_batch, CONFIG["l_auth"], rpp_system,
                        CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
                        size=CONFIG["size"], z_list=CONFIG["z_list"],
                        obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max_rad
                    )
                    p_auth = model(c_auth)

                    # RPP 攻击: 正确 OAM + 错误 RPP
                    rpp_wrong_eval = generate_rpp(CONFIG["size"], device)
                    c_rpp_unauth = encrypt_batch(
                        test_batch, CONFIG["l_auth"], rpp_wrong_eval,
                        CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
                        size=CONFIG["size"], z_list=CONFIG["z_list"],
                        obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max_rad
                    )
                    p_rpp_unauth = model(c_rpp_unauth)

                    # OAM 攻击: 错误 OAM + 正确 RPP (用 l_wrong 全部, 数量需匹配 num_channels)
                    l_wrong_eval = CONFIG["l_wrong"][:num_channels] if len(CONFIG["l_wrong"]) >= num_channels else CONFIG["l_wrong"]
                    c_oam_unauth = encrypt_batch(
                        test_batch, l_wrong_eval, rpp_system,
                        CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
                        size=CONFIG["size"], z_list=CONFIG["z_list"],
                        obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max_rad
                    )
                    p_oam_unauth = model(c_oam_unauth)

                    psnr_list.append(calculate_psnr(p_auth, tgt).item())
                    psnr_center_list.append(calculate_center_psnr(p_auth, tgt).item())  # v2: 中心区 PSNR
                    sr_rpp_list.append(security_ratio(p_auth, p_rpp_unauth).item())
                    sr_oam_list.append(security_ratio(p_auth, p_oam_unauth).item())

            avg_psnr = float(np.mean(psnr_list))
            avg_psnr_center = float(np.mean(psnr_center_list))  # v2: 中心 PSNR
            avg_sr_rpp = float(np.mean(sr_rpp_list))
            avg_sr_oam = float(np.mean(sr_oam_list))
            print(f"Epoch [{epoch}/{CONFIG['epochs']}] | Loss: {epoch_loss:.6f} "
                  f"(auth={epoch_loss_auth:.6f}, xtalk={epoch_loss_xtalk:.6f}, sec={epoch_loss_sec:.6f}) "
                  f"| PSNR: {avg_psnr:.2f} dB | PSNR_C: {avg_psnr_center:.2f} dB "  # v2: 中心 PSNR
                  f"| SR_RPP: {avg_sr_rpp:.4f} | SR_OAM: {avg_sr_oam:.4f}", flush=True)

            # 保存阶段性模型 (含 optimizer/scheduler 状态, 支持断点续训)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'psnr': avg_psnr,                  # 全图 PSNR (可能虚高)
                'psnr_center': avg_psnr_center,    # v2: 中心 PSNR (真实质量)
                'sec_ratio': avg_sr_oam,
                'sr_rpp': avg_sr_rpp,
                'sr_oam': avg_sr_oam,
            }, os.path.join(os.path.dirname(os.path.abspath(__file__)), f"oam_crypt_dnn_epoch_{epoch}.pth"))
            print(f"  [SAVE] Checkpoint saved to oam_crypt_dnn_epoch_{epoch}.pth", flush=True)

            # 缓存最后一个测试样本用于最终可视化
            last_cipher_auth, last_pred_auth, last_tgt = c_auth, p_auth, tgt
            last_cipher_unauth, last_pred_unauth = c_oam_unauth, p_oam_unauth  # 用 OAM 攻击做可视化

    # ---------- 6. 最终可视化 ----------
    print("训练结束，生成最终安全性对比图 final_security_plot.png ...")
    save_security_plot(
        last_cipher_auth, last_pred_auth, last_tgt,
        last_cipher_unauth, last_pred_unauth,
        path="final_security_plot.png"
    )
    print("完成。")
