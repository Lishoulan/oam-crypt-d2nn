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
    # v5 布局模式:
    #   "grid_2x5"    : 10 通道分 2x5 网格放 10 个独立 216x216 区域 (v3/v4 默认, PSNR 30+ dB)
    #   "oam_overlap" : 10 通道全部叠在中心 216x216, 纯靠 OAM+z 分离 (v5/v6 实验, 目标 20+ dB)
    # 切换 layout 时需同步调整: z_list (oam_overlap 用 10cm 间距更稀疏), epochs (50), mid_ch (128), num_layers (3)
    "layout": "oam_overlap",   # v6 active: "grid_2x5" (v4 baseline) / "oam_overlap" (v5/v6 实验)
    "batch_size": 1,           # 批大小 (3层D2NN+安全损失显存大, 降至1防OOM; 梯度累积补回有效批)
    "epochs": 50,              # v6: 5→50 (oam_overlap 模式需更长训练, v5 24 epoch 14.17 dB, v6 目标 20+ dB)
    "lr": 3e-4,                # U-Net 精修层学习率 (Phase 1 修复: 降低 LR 避免损失爆炸)
    "lr_d2nn": 0.05,           # D2NN 物理层学习率 (降低)
    "mid_ch": 48,              # v6 保守: mid_ch 64 在 3层D2NN+U-Net 1080x1080 OOM (8GB GPU 累积 13GB), 48 平衡容量+显存
    "num_layers": 3,           # v6: 2→3 (加深 D2NN, 增强 OAM 解调)
    "freeze_epochs": 0,       # 0=不冻结
    "warmup_epochs": 30,      # v6: 30 (oam_overlap 需要更长 warmup 学物理分离)
    "sec_weight": 0.3,        # v6: 0.0→0.3 (OAM 串扰更强, 必须启用安全损失)
    "xtalk_weight": 0.0,      # 关闭串扰损失 (单通道无串扰)
    "l1_weight": 0.1,         # v4 优化: 启用 L1 损失 0.1 权重(配合 MSE 提升锐度)
    "use_channel_attn": True, # v6 新增: 启用 ChannelAttention (跨通道建模 10 OAM 关系)
    "quick_test_n": 200,     # v6 oam_overlap: 200 样本 (20 batch/epoch, 50 epoch 总 ~2-3 天)

    # v7 算法创新 (突破 oam_overlap 20 dB PSNR 目标,跳出"加大模型"思维定式)
    # 创新 1: Curriculum Learning - 分 4 stage 从易到难渐进训练
    #   关键洞察: 10 通道同时训练梯度互相干扰; 先学 2 通道稳定物理分离, 再扩展
    "curriculum": True,
    "curriculum_stages": [
        # stage 1: 2 通道 (最小,最易分) - l=±25
        {"n_channels": 2, "l_auth": [-25, 25],       "epochs": 6,  "lr": 5e-4,  "z_list": [0.10, 0.55]},
        # stage 2: 5 通道 (奇数非对称) - 加 ±20, ±15
        {"n_channels": 5, "l_auth": [-25, -15, 0, 15, 25],   "epochs": 6, "lr": 4e-4, "z_list": [0.10, 0.20, 0.35, 0.45, 0.55]},
        # stage 3: 8 通道 (原 10 减掉 2 个最难分)
        {"n_channels": 8, "l_auth": [-25, -20, -15, -10, 10, 15, 20, 25], "epochs": 6, "lr": 3e-4, "z_list": [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60]},
        # stage 4: 10 通道 (完整) - 全 10 路
        {"n_channels": 10, "l_auth": [-25, -20, -15, -10, -5, 5, 10, 15, 20, 25], "epochs": 14, "lr": 3e-4, "z_list": [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]},
    ],
    "curriculum_psnr_threshold": 18.0,  # 达到此 PSNR_C 才推进到下一 stage (否则继续当前 stage)

    # 创新 2: Iterative Self-Consistent Refinement - 3-pass 残差自一致
    #   关键洞察: 单次 U-Net forward 一次性映射 30 通道 → 10 通道,信息瓶颈严重;
    #   3 次残差迭代: pass 1 粗定位, pass 2 局部细化, pass 3 物理一致性修正
    #   显存代价: 训练时 U-Net forward ×3, 8GB GPU 在 stage 4 容易 OOM
    #   方案: 默认 False (避免 OOM); 若有 ≥16GB GPU 可设 True
    "iterative_refine": False,
    "n_passes": 3,  # 3 次前向传播, 残差学习 (Pass k 输出 = Pass (k-1) 输出 + U-Net_Δk)
    "iterative_pass_decay": 0.7,  # Pass k 残差缩放 (类似 momentum, 防止过冲)

    # 创新 3: FFT-based OAM Frequency Domain Filter - 频域方位角谐波带通
    #   关键洞察: OAM 拓扑荷 l 对应频域第 l 阶方位角谐波 (angular harmonics);
    #   在 z 平面聚焦后,目标通道的图像信息集中在特定谐波;
    #   对其他通道做带阻滤波, 直接物理性地抑制串扰, 让 U-Net 学得更容易
    "oam_freq_filter": True,
    "oam_filter_bandwidth": 0.15,  # 谐波带通宽度 (相对第 l 阶, 0.15 = ±15%)
    "oam_filter_strength": 0.5,    # 滤波强度 (0=不滤波, 1=完全带阻), 软启动

    # v8 新物理范式 (PolarHNN: Polar Holographic Neural Network)
    # 核心洞察: OAM 拓扑荷 l 是极坐标方位角谐波 exp(ilθ), 笛卡尔卷积不直接捕获此结构
    # 三大创新: PolarConv + OAM-FDD Loss + Multi-scale OAM 频域解码
    #
    # 创新 1: PolarConv (极坐标卷积) - 在 U-Net bottleneck 引入 (r, θ) 极坐标 1D 卷积
    #   物理: 沿 θ 方向 1D 卷积天然对应 OAM 方位角谐波滤波
    #   实施: 笛卡尔 -> 极坐标 (grid_sample) -> 沿 θ + r 1D conv -> 笛卡尔
    "polar_conv": True,             # v8 主开关: 在 UNetRefine bottleneck 加 PolarConv
    "polar_n_r": 32,                # 极坐标 r 方向采样数 (降低避免 OOM)
    "polar_n_theta": 96,           # 极坐标 θ 方向采样数 (OAM l_max=25 需 >50 满足采样定理)
    "polar_theta_kernel": 7,       # θ 方向 1D 卷积核大小
    "polar_init_scale": 0.0,       # 残差缩放初始值 (0=训练初期不扰动)

    # 创新 2: OAM-FDD Loss (OAM 频域正交损失) - 显式约束通道 j 能量集中在 l_j 谐波
    #   物理: 通道 j 解密后应只在第 l_j 阶方位角谐波处有能量, 其他谐波应接近 0
    #   损失: 1 - E_self (最大化自身 l 附近) + λ * E_other (最小化其他 l 附近)
    "oam_fdd_loss": True,          # v8 开关
    "oam_fdd_l_radius": 15,        # 谐波匹配半径 (bin 索引), 1080 W 维下 ±15 bin ≈ 2.8% 带宽
    "oam_fdd_weight": 0.05,        # OAM-FDD 损失权重 (相对 MSE 损失)

    # 创新 3: Multi-scale OAM 频域解码 (在 D2NN 之后, U-Net 之前的多频率分支)
    #   物理: OAM 频率 = l 阶谐波; 不同 l 通道在不同 z 平面聚焦; 多频率并行解码
    #   实施: 提取 OAM 频域 (W 维 FFT) 几个谐波 bin, 加权融合
    "multi_scale_oam": False,      # v8 默认关闭, 显存压力大; 后续 v8.1 启用

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
                  size=1080, z_list=None, obj_encoding="amplitude", theta_max=None,
                  layout="grid_2x5"):
    """
    双密钥加密引擎(支持授权/未授权两种模式，由传入的 oam_keys / rpp 决定)
    - 输入: batch_imgs (B, C, S, S) 明文振幅图 (C = OAM 通道数, S 由调用方决定)
    - 输出: U_cipher (B, size, size) 复数密文场

    v5 布局模式 (layout):
      - "grid_2x5"    : 2x5 网格, 每通道 216x216 放独立位置 (v3/v4 默认行为)
      - "oam_overlap" : 全部 10 通道叠加在中心 216x216, 纯靠 OAM+z 分离 (v5 实验)

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

    if layout == "grid_2x5":
        # v3/v4 行为: 2x5 网格布局, 每格 216x216, 整体 432x1080 居中
        rows, cols = 2, 5
        cell_h, cell_w = half, half
        y_start = (size - rows * cell_h) // 2
        x_start = (size - cols * cell_w) // 2
        positions = []
        for i in range(len(oam_keys)):
            r = i // cols
            c = i % cols
            y = y_start + r * cell_h
            x = x_start + c * cell_w
            positions.append((y, x))
    elif layout == "oam_overlap":
        # v5 新增: 全部 10 通道叠在中心 216x216 同一位置, 纯 OAM + z 分离
        y_start = (size - half) // 2  # 432
        x_start = (size - half) // 2  # 432
        positions = [(y_start, x_start)] * len(oam_keys)
    else:
        raise ValueError(f"Unknown layout: {layout}, expect 'grid_2x5' or 'oam_overlap'")

    for i, l in enumerate(oam_keys):
        img_pad = torch.zeros(B, size, size, device=device)
        y, x = positions[i]
        img_pad[:, y:y+half, x:x+half] = batch_imgs[:, i]

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


def build_target_grid(batch_imgs, device, size=1080, num_channels=None, layout="grid_2x5"):
    """
    构建 (B, C, size, size) 目标网格: 每通道对应一个图像
    v5 布局模式 (layout):
      - "grid_2x5"    : 2x5 网格, 每通道放独立 216x216 位置 (v3/v4 行为)
      - "oam_overlap" : 全部通道都放中心 216x216 同一位置 (v5 实验)
    C 由 num_channels 决定 (默认从 batch_imgs 自动推断, 与 OAM 通道数一致)
    """
    B, C, H, W = batch_imgs.shape
    if num_channels is None:
        num_channels = C
    target = torch.zeros(B, num_channels, size, size, device=device)

    if layout == "grid_2x5":
        # 2x5 网格布局 (10 通道 v3/v4 默认)
        rows, cols = 2, 5
        y_start = (size - rows * H) // 2
        x_start = (size - cols * W) // 2
        for i in range(num_channels):
            r = i // cols
            c = i % cols
            y = y_start + r * H
            x = x_start + c * W
            target[:, i, y:y+H, x:x+W] = batch_imgs[:, i]
    elif layout == "oam_overlap":
        # v5 新增: 全部通道叠在中心 216x216
        y_start = (size - H) // 2
        x_start = (size - W) // 2
        for i in range(num_channels):
            target[:, i, y_start:y_start+H, x_start:x_start+W] = batch_imgs[:, i]
    else:
        raise ValueError(f"Unknown layout: {layout}, expect 'grid_2x5' or 'oam_overlap'")
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

class OAMFreqFilter(nn.Module):
    """
    v7 创新 3: FFT-based OAM Frequency Domain Filter (频域方位角谐波带通/带阻)

    核心原理:
      OAM 拓扑荷 l 对应光场在极坐标下方位角 θ 的第 l 阶谐波 exp(ilθ)。
      离散化: 圆环上第 r 行像素的相位随 θ 绕 2π 变化 l 次。
      在频域,这种空间结构 → 在方位角维度的 FFT 中,峰值出现在第 l 个频率 bin。

    应用:
      对 U_back 的每个 OAM 通道 j, 在反向传播 (ASM(-z_j)) 之后做频域滤波:
      - 通道 j 自身: 中心谐波 (第 l_j 阶) 保留, 其他谐波按带宽带阻
      - 物理效果: 抑制来自其他 OAM 通道的串扰, 让 U-Net 学得更容易

    实现:
      对每行 r 像素, 在 W 维度 (或等价的极坐标 θ 维度) 做 FFT,
      在频率维度根据各通道 l_j 做带通/带阻滤波 (高斯带阻 mask)。
      因为已做反向传播聚焦, 大部分信号能量集中在低频附近,
      高频部分主要是 OAM 串扰和噪声。

    Args:
        oam_keys: list of int, 长度 num_channels, 各通道的 l 拓扑荷
        size: 空间尺寸 (H, W)
        bandwidth: 相对带宽 (相对第 l 阶谐波, 0.15 = ±15%)
        strength: 滤波强度 (0=不滤波, 1=完全带阻)
    """
    def __init__(self, oam_keys, size, bandwidth=0.15, strength=0.5):
        super().__init__()
        self.oam_keys = oam_keys
        self.size = size
        self.bandwidth = bandwidth
        self.strength = strength
        # 预计算每通道的频域 mask (num_channels, W/2+1 频域 bins)
        # 简化: OAM 谐波频率 ≈ |l| 阶; 通道 j 的目标频率 bin = |l_j|
        # mask: 目标 bin 附近 ± bandwidth*W/2 通过, 其他按 strength 衰减
        n_freq = size // 2 + 1
        masks = torch.ones(len(oam_keys), n_freq)
        for j, l in enumerate(oam_keys):
            # OAM l 在 W 维度的目标谐波 bin ≈ |l| (假设每行 360° 对应 W 个像素, 谐波 bin 索引 l)
            target_bin = min(abs(l), n_freq - 1)
            # 软带通: 中心 1.0, 远离处按 (1-strength) 衰减
            for k in range(n_freq):
                dist = abs(k - target_bin) / max(n_freq, 1)
                if dist < bandwidth:
                    masks[j, k] = 1.0  # 通过
                else:
                    masks[j, k] = 1.0 - strength  # 衰减
        # 复数频域需要双边 (W bins, 含负频率), 镜像扩展
        mask_full = torch.cat([masks, masks.flip(1)[:, 1:-1]], dim=1)  # (C, W)
        self.register_buffer('mask', mask_full)  # (C, W)

    def forward(self, x_complex):
        """
        Args:
            x_complex: (B, C, H, W) complex, 各 OAM 通道在 z_j 平面聚焦后的复振幅
        Returns:
            x_filtered: (B, C, H, W) complex, 频域滤波后
        """
        B, C, H, W = x_complex.shape
        # 对 W 维度做 FFT
        x_fft = torch.fft.fft(x_complex, dim=-1)  # (B, C, H, W) complex
        # 应用 mask
        mask = self.mask.unsqueeze(0).unsqueeze(2)  # (1, C, 1, W)
        x_fft_filt = x_fft * mask  # 复数乘法
        # 逆变换
        x_filtered = torch.fft.ifft(x_fft_filt, dim=-1)
        return x_filtered


class PolarConv(nn.Module):
    """
    v8 新物理范式 1: PolarConv (极坐标卷积)

    物理动机:
      OAM 拓扑荷 l 是光场在极坐标 (r, θ) 中方位角 θ 的第 l 阶谐波
      exp(ilθ) 沿 θ 方向有 l 个周期。笛卡尔卷积 (3×3) 不直接捕获此
      方位角结构, 因为卷积核对方位角方向不敏感。

    实施:
      1. 笛卡尔 -> 极坐标: 双线性插值 (grid_sample) 从 (H, W) 笛卡尔网格
         采样到 (n_r, n_theta) 极坐标网格
      2. 沿 θ 方向 1D 深度可分离卷积: 等价于在方位角谐波空间滤波
         kernel_size=7 覆盖约 7 个相邻谐波 bin
      3. 沿 r 方向 1D 深度可分离卷积: 捕获径向结构
      4. 极坐标 -> 笛卡尔: 反向 grid_sample 回到 (H, W) 笛卡尔网格
      5. 残差连接 + scale 缩放 (init=0 训练初期不扰动)

    显存优化:
      - n_r=32, n_theta=96: 总采样点 3072 (vs 270×270=72900), 大幅降低
      - 极坐标特征 (B, C, 32, 96) 而非 (B, C, 270, 270), 节省 ~7× 显存
      - 深度可分离卷积 (groups=C): 进一步降低参数量和计算量

    放置位置: UNetRefine bottleneck 处 (270×270, mid_ch*4=192 通道)
    """
    def __init__(self, channels, n_r=32, n_theta=96, theta_kernel=7, init_scale=0.0):
        super().__init__()
        self.channels = channels
        self.n_r = n_r
        self.n_theta = n_theta
        # 沿 θ 方向 1D 深度可分离卷积 (捕获 OAM 方位角谐波)
        self.theta_conv = nn.Conv1d(channels, channels, kernel_size=theta_kernel,
                                     padding=theta_kernel // 2, groups=channels, bias=False)
        self.theta_pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        # 沿 r 方向 1D 深度可分离卷积 (捕获径向结构)
        self.r_conv = nn.Conv1d(channels, channels, kernel_size=3,
                                padding=1, groups=channels, bias=False)
        self.r_pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        # 残差缩放 (init=0 训练初期等于恒等)
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        # 极坐标 -> 笛卡尔 笛卡尔坐标索引缓存 (延迟到 forward 第一次创建, 跟随 device)

    def _make_polar_sample_grid(self, H, W, device):
        """生成 (n_r, n_theta, 2) 极坐标 -> 笛卡尔 grid, 笛卡尔 grid 归一化到 [-1,1]"""
        cx, cy = W / 2.0, H / 2.0
        R_max = min(cx, cy)
        r = torch.linspace(0, R_max, self.n_r, device=device)
        theta = torch.linspace(0, 2 * np.pi, self.n_theta, device=device)
        r_grid, theta_grid = torch.meshgrid(r, theta, indexing='ij')  # (n_r, n_theta)
        x_polar = cx + r_grid * torch.cos(theta_grid)
        y_polar = cy + r_grid * torch.sin(theta_grid)
        x_norm = 2 * x_polar / (W - 1) - 1
        y_norm = 2 * y_polar / (H - 1) - 1
        grid = torch.stack([x_norm, y_norm], dim=-1)  # (n_r, n_theta, 2)
        return grid

    def _make_full_polar_grid(self, H, W, device):
        """生成 (H, W, 2) 笛卡尔 -> 极坐标 grid, 归一化到 [-1,1]"""
        cx, cy = W / 2.0, H / 2.0
        R_max = min(cx, cy)
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )  # (H, W)
        r_full = torch.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        theta_full = torch.atan2(y_grid - cy, x_grid - cx) % (2 * np.pi)
        r_idx = 2 * (r_full / R_max).clamp(0, 1) - 1
        theta_idx = 2 * (theta_full / (2 * np.pi)) - 1
        grid = torch.stack([r_idx, theta_idx], dim=-1)  # (H, W, 2) (x=r, y=theta)
        return grid

    def forward(self, x):
        """
        x: (B, C, H, W) 笛卡尔坐标特征
        """
        B, C, H, W = x.shape
        device = x.device

        # 1. 笛卡尔 -> 极坐标: grid_sample
        polar_sample_grid = self._make_polar_sample_grid(H, W, device)
        polar_sample_grid = polar_sample_grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, n_r, n_theta, 2)
        x_p = F.grid_sample(x, polar_sample_grid, mode='bilinear',
                             align_corners=False, padding_mode='zeros')  # (B, C, n_r, n_theta)

        # 2. 沿 θ 方向 1D 深度可分离卷积 (n_theta 维度)
        #    Conv1d 期望 (B, C_in, L_in) 格式
        #    重塑 (B, C, n_r, n_theta) -> (B*n_r, C, n_theta) 沿 n_theta 卷积
        #    groups=C 深度可分离, 每个通道独立沿 θ 卷积
        x_p_flat = x_p.permute(0, 2, 1, 3).reshape(B * self.n_r, C, self.n_theta)  # (B*n_r, C, n_theta)
        x_p_flat = self.theta_conv(x_p_flat)  # 沿最后维 (n_theta) 卷积, groups=C
        x_p_flat = self.theta_pointwise(x_p_flat)  # 1x1 通道混合
        # 还原 (B, C, n_r, n_theta)
        x_p = x_p_flat.reshape(B, self.n_r, C, self.n_theta).permute(0, 2, 1, 3).contiguous()

        # 3. 沿 r 方向 1D 深度可分离卷积
        #    重塑 (B, C, n_r, n_theta) -> (B*n_theta, C, n_r) 沿 n_r 卷积
        x_p_flat = x_p.permute(0, 3, 1, 2).reshape(B * self.n_theta, C, self.n_r)  # (B*n_theta, C, n_r)
        x_p_flat = self.r_conv(x_p_flat)  # 沿最后维 (n_r) 卷积, groups=C
        x_p_flat = self.r_pointwise(x_p_flat)
        # 还原
        x_p = x_p_flat.reshape(B, self.n_theta, C, self.n_r).permute(0, 2, 3, 1).contiguous()  # (B, C, n_r, n_theta)

        # 4. 极坐标 -> 笛卡尔
        full_polar_grid = self._make_full_polar_grid(H, W, device)
        full_polar_grid = full_polar_grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
        x_out = F.grid_sample(x_p, full_polar_grid, mode='bilinear',
                                align_corners=False, padding_mode='zeros')  # (B, C, H, W)

        # 5. 残差 + scale 缩放
        return x + self.scale * (x_out - x)


class ChannelAttention(nn.Module):
    """
    v6 新增: 跨通道注意力 (Squeeze-and-Excitation 风格)
    显式建模 10 通道 OAM 关系, 让网络自适应调整各通道权重。
    关键: OAM 通道正交性让各通道理论独立, 但 OAM+z 重叠模式
    10 通道在中心 216×216 同位置, 信道间存在串扰;
    ChannelAttention 让网络自动学习"哪些通道当前需要加强/抑制"。

    Args:
        num_channels: 输入通道数 (v6: 3 * num_oam_channels = 30)
        reduction: 压缩比 (默认 4)
    """
    def __init__(self, num_channels, reduction=4):
        super().__init__()
        mid = max(num_channels // reduction, 8)
        self.fc1 = nn.Conv2d(num_channels, mid, 1, bias=False)
        self.fc2 = nn.Conv2d(mid, num_channels, 1, bias=False)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - 30 通道 (10 OAM × 3 features)
        Returns:
            x * channel_weight: (B, C, H, W)
        """
        w = self.gap(x)            # (B, C, 1, 1) 全局平均池化
        w = F.relu(self.fc1(w), inplace=True)  # 压缩
        w = self.sigmoid(self.fc2(w))           # 激励
        return x * w


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

    v6 新增: 可选 ChannelAttention (use_channel_attn=True) 显式建模跨 OAM 通道关系
    v8 新增: 可选 PolarConv (use_polar_conv=True) 在 e3 -> bot 之间引入极坐标卷积
            物理: 极坐标 (r, θ) 卷积天然捕获 OAM 方位角谐波结构
    """
    def __init__(self, in_ch=1, out_ch=1, mid_ch=64, use_channel_attn=True,
                 use_polar_conv=False, polar_n_r=32, polar_n_theta=96,
                 polar_theta_kernel=7, polar_init_scale=0.0):
        super().__init__()

        # v6 新增: 跨通道注意力 (30 通道 = 10 OAM × 3 features)
        self.channel_attn = ChannelAttention(in_ch) if use_channel_attn else nn.Identity()

        # Encoder
        self.enc1 = self._conv_block(in_ch, mid_ch)
        self.enc2 = self._conv_block(mid_ch, mid_ch * 2)
        self.enc3 = self._conv_block(mid_ch * 2, mid_ch * 4)

        # v8 新增: PolarConv (e3 -> bot 之间, 在 mid_ch*4 通道空间)
        #   物理: e3 输出 (B, 4*mid, H/4, W/4) 在 1080×1080 输入时是 270×270
        #   极坐标卷积对 OAM 方位角谐波结构敏感, 直接增强 OAM 通道分离
        self.polar_conv = (PolarConv(mid_ch * 4, n_r=polar_n_r, n_theta=polar_n_theta,
                                      theta_kernel=polar_theta_kernel, init_scale=polar_init_scale)
                           if use_polar_conv else nn.Identity())

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

        # v6 新增: 跨通道注意力 (在 U-Net encoder 之前)
        x_norm = self.channel_attn(x_norm)

        e1 = self.enc1(x_norm)                              # (B, mid, H, W)
        e2 = self.enc2(self._down(e1))                      # (B, 2mid, H/2, W/2)
        e3 = self.enc3(self._down(e2))                     # (B, 4mid, H/4, W/4)
        # v8 新增: 极坐标卷积 (e3 -> bot 之间)
        e3 = self.polar_conv(e3)
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
                 obj_encoding="amplitude", theta_max=None, slm_aware=True,
                 use_channel_attn=True, mid_ch=None,
                 iterative_refine=True, n_passes=3, iterative_pass_decay=0.7,
                 oam_freq_filter=True, oam_filter_bandwidth=0.15, oam_filter_strength=0.5,
                 use_polar_conv=False, polar_n_r=32, polar_n_theta=96,
                 polar_theta_kernel=7, polar_init_scale=0.0):
        super().__init__()
        self.layers = nn.ModuleList([DiffractiveLayer(size, nonlin_every=3, layer_idx=i) for i in range(num_layers)])
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.z_layer = z_layer
        self.z0 = z0
        self.obj_encoding = obj_encoding
        self.theta_max = theta_max  # k 空间约束 (弧度), None=不约束
        self.slm_aware = slm_aware  # True=在 forward 内部模拟 SLM 8-bit 量化
        # v7 创新 2: Iterative Refinement
        self.iterative_refine = iterative_refine
        self.n_passes = n_passes
        self.iterative_pass_decay = iterative_pass_decay
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
        # v7 创新 3: OAM 频域滤波器
        if oam_freq_filter and oam_keys is not None:
            self.oam_filter = OAMFreqFilter(oam_keys, size, oam_filter_bandwidth, oam_filter_strength)
        else:
            self.oam_filter = None
        # out_ch = num_channels: 每个通道输出一个图像, 都在同一位置 (多平面复用)
        # v2: 3 通道/图像 (|U|, real, imag), 总输入 = 3 * num_channels
        # v6: mid_ch 可外部覆盖, 默认从 CONFIG 取; use_channel_attn 启用跨通道建模
        # v8: use_polar_conv 启用 UNetRefine bottleneck 极坐标卷积
        _mid_ch = mid_ch if mid_ch is not None else CONFIG["mid_ch"]
        self.refine = UNetRefine(
            in_ch=3 * self.num_channels, out_ch=self.num_channels,
            mid_ch=_mid_ch, use_channel_attn=use_channel_attn,
            use_polar_conv=use_polar_conv,
            polar_n_r=polar_n_r, polar_n_theta=polar_n_theta,
            polar_theta_kernel=polar_theta_kernel, polar_init_scale=polar_init_scale,
        )
        # v7 创新 2: Iterative Refinement 用 1x1 conv 把 refined (C 通道) 反馈到 3C 通道空间
        # 新增参数: C * 3C = 3C² (C=10 时 300 参数, 可忽略)
        # 总是创建 (即使 iterative_refine=False), 保证 state_dict 跨版本兼容
        self.context_proj = nn.Conv2d(self.num_channels, 3 * self.num_channels, kernel_size=1)

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

        # v7 创新 3: OAM 频域滤波 (在 D2NN 之前抑制跨通道谐波串扰)
        # 把 (B*C, H, W) reshape 回 (B, C, H, W) 做滤波, 再 reshape 回扁平
        if self.oam_filter is not None:
            U_back_bc = U_back.reshape(B, self.num_channels, U.shape[-2], U.shape[-1])
            U_back_bc = self.oam_filter(U_back_bc)  # 复数域带阻滤波
            U_back = U_back_bc.reshape(B * self.num_channels, U.shape[-2], U.shape[-1])

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

        # v7 创新 2: Iterative Self-Consistent Refinement (3-pass 残差自一致)
        # Pass 1: 粗定位 refined = U_Net(x)
        # Pass k (k>=2): feedback = 1x1_conv(refined) 投影到 3C 空间叠加到 x, 学 Δ
        # 训练时启用; 推理时用单 pass 加速
        refined = self.refine(x)  # (B, C, H, W) Pass 1
        if self.iterative_refine and self.context_proj is not None and self.n_passes > 1:
            n_extra = self.n_passes - 1
            # 训练时启用 iterative; 推理保持单 pass (节省时间)
            if self.training:
                for k in range(1, n_extra + 1):
                    decay_k = self.iterative_pass_decay ** k
                    feedback = self.context_proj(refined)  # (B, 3C, H, W)
                    x_iter = x + decay_k * feedback
                    delta = self.refine(x_iter)  # 共享 U-Net, 学 Δ 残差
                    refined = refined + decay_k * delta

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
    v5 oam_overlap: 中心区域改为 216x216 居中方块 (y=[432:648], x=[432:648])
        旧 center_size=270 已被忽略, 根据 CONFIG["layout"] 自适应
    pred: (B, C, H, W) 解密图像
    target: (B, C, H, W) 明文图像 (build_target_grid 输出)
    旧 PSNR 在 1080×1080 全图算 (94% 是黑边), 虚高; 中心 PSNR 反映肉眼可见的质量
    """
    pred = pred.clamp(0, 1)
    # 根据布局模式自适应中心区域
    if CONFIG.get("layout", "grid_2x5") == "oam_overlap":
        # v5: 全部通道在中心 216x216 (y=[432:648], x=[432:648])
        pred_c = pred[..., 432:648, 432:648]
        tgt_c = target[..., 432:648, 432:648]
    else:
        # v3/v4 默认: 2x5 网格覆盖范围 y=[324:756] (432 高), x=[0:1080] (整宽)
        pred_c = pred[..., 324:756, 0:1080]
        tgt_c = target[..., 324:756, 0:1080]
    mse = torch.mean((pred_c - tgt_c) ** 2)
    if mse <= 0:
        return torch.tensor(float('inf'), device=mse.device)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def oam_fdd_loss(pred, oam_keys, l_radius=15, size=1080):
    """
    v8 新物理范式 2: OAM-FDD Loss (OAM 频域正交损失)

    物理动机:
      OAM 拓扑荷 l_j 通道解密后, 其图像信息应集中在方位角 θ 的第 l_j 阶谐波
      频域 bin |l_j| 附近 (因 exp(ilθ) 沿 θ 周期 l 次)。
      不同通道的谐波能量应正交不重叠, 即通道 j 能量集中在 bin |l_j| 附近,
      远离其他通道 |l_i| (i≠j) 区域。

    实施 (v8 改进版: 通道间频域相关性矩阵):
      1. 对每通道, 沿 W 维度 (近似方位角方向) 做 FFT, 得到频域 Y_j(k)
      2. 归一化每个通道: Y_norm_j = Y_j / ||Y_j||  (单位频域向量)
      3. 通道间频域相关性矩阵: R_ij = |<Y_i, Y_j>| = |Σ_k Y_i(k) * Y_j(k)*|
         理想情况: R_ij = δ_ij (完全正交)
      4. 损失: Σ_{i≠j} R_ij² (最小化通道间相关性)

    物理效果:
      - 直接强制 OAM 通道在频域正交, 化解"同位置重叠"串扰
      - 配合 PolarConv: PolarConv 增强 θ 方向结构, FDD Loss 强化频域正交
      - 不依赖 zone 范围 (l_radius 仅用于诊断, 损失用全频域归一化)

    Args:
        pred: (B, C, H, W) U-Net 输出 (clamp 后)
        oam_keys: list of int, 各通道 l 拓扑荷
        l_radius: 谐波匹配半径 (用于诊断, 损失不依赖)
        size: 空间尺寸 (默认 1080)
    Returns:
        loss: scalar tensor
    """
    B, C, H, W = pred.shape
    device = pred.device
    # 关键: 强制 float32, 因为 cuFFT 在 half 精度下不支持非 2 幂维度 (如 1080)
    pred = pred.float()
    # 1. 沿 W 维度 FFT
    Y = torch.fft.fft(pred, dim=-1)  # (B, C, H, W) complex
    # 2. 排除 DC bin (bin 0) 和镜像 bin (bin W/2), 关注 OAM 谐波部分
    #    图像能量大部分在 DC, 会让所有通道相关性都接近 1 (DC 共享)
    Y_no_dc = Y.clone()
    Y_no_dc[..., 0] = 0
    if W % 2 == 0:
        Y_no_dc[..., W // 2] = 0  # Nyquist bin
    # 3. L2 归一化
    Y_norm = Y_no_dc / (Y_no_dc.norm(dim=-1, keepdim=True) + 1e-8)  # (B, C, H, W)
    # 4. 通道间点积矩阵: R[i,j] = <Y_i, Y_j> = Σ_k Y_i(k) * Y_j(k)*
    R = torch.einsum('bchk,bjhk->bchj', Y_norm, Y_norm.conj())  # (B, C, H, C)
    R = R.abs()  # 标量
    # 5. 提取上三角 (i<j) 通道间相关性
    mask = torch.triu(torch.ones(C, C, device=device), diagonal=1)
    R_offdiag = R * mask.unsqueeze(0).unsqueeze(2)  # (B, C, H, C)
    n_pairs = mask.sum().item()
    # 6. 损失: 平均通道间相关性 (理想正交时为 0)
    loss = R_offdiag.sum() / (n_pairs * B * H + 1e-8)
    return loss


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


def viz_decrypted_grid(pred, save_path="decrypted_grid_2x5.png"):
    """
    v5 新增: 把 10 通道解密输出 (B, 10, 1080, 1080) 拼成 2x5 网格图
    每通道裁剪中心 216x216, 横向拼接成 2 行 (432x1080), 便于一眼看 10 个数字
    不管 layout 是 grid_2x5 还是 oam_overlap, 都能直观展示 10 路结果
    """
    import matplotlib.pyplot as plt
    from font_config import setup_cjk
    setup_cjk()

    C = pred.shape[1]
    cy = pred.shape[-2] // 2
    cx = pred.shape[-1] // 2
    half = 108  # 216/2

    cells = []
    for i in range(C):
        cell = pred[0, i, cy-half:cy+half, cx-half:cx+half].detach().cpu().numpy()
        cells.append(np.clip(cell, 0, 1))

    if C == 10:
        # 2 行 x 5 列布局
        row0 = np.concatenate(cells[:5], axis=1)  # 216 x 1080
        row1 = np.concatenate(cells[5:], axis=1)  # 216 x 1080
        grid = np.concatenate([row0, row1], axis=0)  # 432 x 1080
    else:
        # 自适应: 找接近正方形的网格
        cols = int(np.ceil(np.sqrt(C)))
        rows = int(np.ceil(C / cols))
        rows_imgs = []
        for r in range(rows):
            row_cells = cells[r*cols:(r+1)*cols]
            if len(row_cells) < cols:
                # 不足补黑色
                pad = [np.zeros_like(cells[0])] * (cols - len(row_cells))
                row_cells = row_cells + pad
            rows_imgs.append(np.concatenate(row_cells, axis=1))
        grid = np.concatenate(rows_imgs, axis=0)

    fig, ax = plt.subplots(figsize=(15, 3))
    ax.imshow(grid, cmap='gray', vmin=0, vmax=1, aspect='equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"v5 解密结果 2x5 网格拼图 (10 通道, layout={CONFIG.get('layout', 'grid_2x5')})",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [VIZ] 已保存: {save_path} ({grid.shape[0]}x{grid.shape[1]})")


# ==========================================
# 阶段7：主训练与测试流程
# ==========================================

def train_one_stage(stage_cfg, stage_idx, n_stages, device, rpp_system,
                    quick_test_n=200, num_layers=3, mid_ch=48,
                    use_channel_attn=True, layout="oam_overlap",
                    verbose=True):
    """
    v7 新增: 单个 curriculum stage 的训练函数 (支持 l_auth/z_list 动态变化)
    关键: l_auth 改变 → OAM 堆栈、U-Net 通道、OAMFreqFilter mask 全部变化 → 必须重建模型

    Args:
        stage_cfg: {"n_channels", "l_auth", "epochs", "lr", "z_list"}
        stage_idx: 当前 stage 编号 (0-based)
        n_stages: 总 stage 数
        device: torch device
        rpp_system: 授权 RPP (跨 stage 共享)
        quick_test_n: 数据子集大小
        num_layers, mid_ch, use_channel_attn, layout: 模型超参

    Returns:
        best_psnr_center, best_model_state, best_sec_ratio
    """
    l_auth_stage = stage_cfg["l_auth"]
    z_list_stage = stage_cfg["z_list"]
    n_ch_stage = stage_cfg["n_channels"]
    epochs_stage = stage_cfg["epochs"]
    lr_stage = stage_cfg["lr"]

    print(f"\n{'='*80}\n"
          f"[CURRICULUM STAGE {stage_idx+1}/{n_stages}] n_ch={n_ch_stage} "
          f"l_auth={l_auth_stage} z_list={z_list_stage}\n"
          f"  epochs={epochs_stage} lr={lr_stage}\n{'='*80}", flush=True)

    # 数据准备
    n_train = quick_test_n if quick_test_n else 1600
    n_test = quick_test_n if quick_test_n else 400
    transform = transforms.Compose([transforms.ToTensor()])
    full_train = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    full_test = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    mnist_train = Subset(full_train, range(n_train))
    mnist_test = Subset(full_test, range(n_test))

    img_size = CONFIG["size"] // 5
    train_dataset = MNISTQuadDataset(mnist_train, img_size=img_size, num_channels=n_ch_stage)
    test_dataset = MNISTQuadDataset(mnist_test, img_size=img_size, num_channels=n_ch_stage)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    # 重建模型
    theta_max_rad = np.deg2rad(CONFIG["theta_max_deg"]) if CONFIG.get("theta_max_deg") else None
    model = OAM_Crypt_D2NN(
        size=CONFIG["size"], num_layers=num_layers,
        wavelength=CONFIG["wavelength"], pixel_size=CONFIG["pixel_size"],
        z_layer=CONFIG["z_layer"], z0=CONFIG["z0"], rpp=rpp_system,
        oam_keys=l_auth_stage, z_list=z_list_stage,
        obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max_rad,
        slm_aware=CONFIG["slm_aware"],
        use_channel_attn=use_channel_attn,
        mid_ch=mid_ch,
        iterative_refine=CONFIG.get("iterative_refine", True),
        n_passes=CONFIG.get("n_passes", 3),
        iterative_pass_decay=CONFIG.get("iterative_pass_decay", 0.7),
        oam_freq_filter=CONFIG.get("oam_freq_filter", True),
        oam_filter_bandwidth=CONFIG.get("oam_filter_bandwidth", 0.15),
        oam_filter_strength=CONFIG.get("oam_filter_strength", 0.5),
        use_polar_conv=CONFIG.get("polar_conv", True),
        polar_n_r=CONFIG.get("polar_n_r", 32),
        polar_n_theta=CONFIG.get("polar_n_theta", 96),
        polar_theta_kernel=CONFIG.get("polar_theta_kernel", 7),
        polar_init_scale=CONFIG.get("polar_init_scale", 0.0),
    ).to(device)

    with torch.no_grad():
        for i in range(num_layers):
            model.layers[i].phase.zero_()
            model.layers[i].amp_logit.fill_(4.0)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [STAGE {stage_idx+1}] 模型已重建, 可训练参数={n_params:,}", flush=True)

    # 优化器
    d2nn_params = list(model.layers.parameters())
    refine_params = list(model.refine.parameters())
    param_groups = [{"params": refine_params, "lr": lr_stage}]
    if len(d2nn_params) > 0:
        param_groups.insert(0, {"params": d2nn_params, "lr": CONFIG["lr_d2nn"]})
    optimizer = optim.Adam(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_stage)
    criterion_mse = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # 错误 OAM 列表 (数量匹配 stage n_channels)
    l_wrong_stage = [l + 1 if l >= 0 else l - 1 for l in l_auth_stage]
    if len(CONFIG["l_wrong"]) >= n_ch_stage:
        l_wrong_stage = CONFIG["l_wrong"][:n_ch_stage]

    best_psnr_center = -float('inf')
    best_model_state = None
    best_sr_oam = None

    # 训练循环
    for epoch in range(1, epochs_stage + 1):
        model.train()
        epoch_loss = epoch_loss_auth = epoch_loss_sec = epoch_loss_fdd = 0.0
        n_batches = len(train_loader)
        t_start = __import__("time").time()
        use_security = epoch > max(1, int(epochs_stage * 0.5))

        for bidx, batch_imgs in enumerate(train_loader):
            batch_imgs = batch_imgs.to(device)
            target = build_target_grid(batch_imgs, device, size=CONFIG["size"], layout=layout)

            cipher_auth = encrypt_batch(
                batch_imgs, l_auth_stage, rpp_system,
                CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
                size=CONFIG["size"], z_list=z_list_stage,
                obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max_rad,
                layout=layout
            )

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pred_auth = model(cipher_auth)

                H, W = target.shape[-2:]
                weight_map = torch.ones(1, 1, H, W, device=device) * 0.1
                if layout == "oam_overlap":
                    weight_map[..., 432:648, 432:648] = 10.0
                else:
                    weight_map[..., 324:756, 0:1080] = 10.0
                loss_mse = torch.mean(weight_map * (pred_auth - target) ** 2)
                loss_l1 = torch.mean(weight_map * torch.abs(pred_auth - target))
                loss_auth = loss_mse + 0.1 * loss_l1

                loss_sec = torch.tensor(0.0, device=device)
                if use_security and bidx % 2 == 0:
                    cipher_unauth = encrypt_batch(
                        batch_imgs, l_wrong_stage, rpp_system,
                        CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
                        size=CONFIG["size"], z_list=z_list_stage,
                        obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max_rad,
                        layout=layout
                    )
                    pred_unauth = model(cipher_unauth)
                    loss_sec = criterion_mse(pred_unauth, torch.zeros_like(pred_unauth))

                # v8 创新 2: OAM-FDD 频域正交损失
                # 配合 PolarConv 强化 OAM 通道频域正交性
                loss_fdd = torch.tensor(0.0, device=device)
                if CONFIG.get("oam_fdd_loss", True) and epoch > max(1, int(epochs_stage * 0.3)):
                    pred_clamped = pred_auth.clamp(0, 1)
                    loss_fdd = oam_fdd_loss(
                        pred_clamped, l_auth_stage,
                        l_radius=CONFIG.get("oam_fdd_l_radius", 3),
                        size=CONFIG["size"],
                    )

                total_loss = (loss_auth + 0.3 * loss_sec
                              + CONFIG.get("oam_fdd_weight", 0.05) * loss_fdd)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = batch_imgs.size(0)
            epoch_loss += total_loss.item() * bs
            epoch_loss_auth += loss_auth.item() * bs
            epoch_loss_sec += loss_sec.item() * bs
            epoch_loss_fdd += loss_fdd.item() * bs

        scheduler.step()
        n = len(train_loader.dataset)
        epoch_loss /= n; epoch_loss_auth /= n; epoch_loss_sec /= n
        epoch_loss_fdd /= n

        # 验证
        model.eval()
        psnr_c_list = []; sr_rpp_list = []; sr_oam_list = []
        with torch.no_grad():
            for test_batch in test_loader:
                test_batch = test_batch.to(device)
                tgt = build_target_grid(test_batch, device, size=CONFIG["size"], layout=layout)
                c_auth = encrypt_batch(test_batch, l_auth_stage, rpp_system,
                    CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
                    size=CONFIG["size"], z_list=z_list_stage,
                    obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max_rad, layout=layout)
                p_auth = model(c_auth)
                rpp_w = generate_rpp(CONFIG["size"], device)
                c_rpp = encrypt_batch(test_batch, l_auth_stage, rpp_w,
                    CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
                    size=CONFIG["size"], z_list=z_list_stage,
                    obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max_rad, layout=layout)
                p_rpp = model(c_rpp)
                c_oam = encrypt_batch(test_batch, l_wrong_stage, rpp_system,
                    CONFIG["z0"], CONFIG["wavelength"], CONFIG["pixel_size"], device,
                    size=CONFIG["size"], z_list=z_list_stage,
                    obj_encoding=CONFIG["obj_encoding"], theta_max=theta_max_rad, layout=layout)
                p_oam = model(c_oam)
                psnr_c_list.append(calculate_center_psnr(p_auth, tgt).item())
                sr_rpp_list.append(security_ratio(p_auth, p_rpp).item())
                sr_oam_list.append(security_ratio(p_auth, p_oam).item())
        avg_psnr_c = float(np.mean(psnr_c_list))
        avg_sr_rpp = float(np.mean(sr_rpp_list))
        avg_sr_oam = float(np.mean(sr_oam_list))
        elapsed = __import__("time").time() - t_start
        print(f"  [S{stage_idx+1} E{epoch}/{epochs_stage}] loss={epoch_loss:.5f} "
              f"(auth={epoch_loss_auth:.5f}, sec={epoch_loss_sec:.5f}, fdd={epoch_loss_fdd:.5f}) "
              f"PSNR_C={avg_psnr_c:.2f} dB SR_RPP={avg_sr_rpp:.4f} SR_OAM={avg_sr_oam:.4f} "
              f"t={elapsed:.0f}s", flush=True)

        if avg_psnr_c > best_psnr_center:
            best_psnr_center = avg_psnr_c
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_sr_oam = avg_sr_oam
            print(f"    [BEST] PSNR_C {best_psnr_center:.2f} dB (epoch {epoch})", flush=True)

    return best_psnr_center, best_model_state, best_sr_oam


if __name__ == "__main__":
    device = torch.device(CONFIG["device"])
    torch.manual_seed(42)
    np.random.seed(42)

    # ---------- 0. layout 自适应参数覆盖 ----------
    if CONFIG["layout"] == "oam_overlap":
        CONFIG["epochs"] = max(CONFIG.get("epochs", 0), 50)
        CONFIG["warmup_epochs"] = max(CONFIG.get("warmup_epochs", 0), 30)
        CONFIG["sec_weight"] = 0.3
        CONFIG["mid_ch"] = 48
        CONFIG["num_layers"] = 3
        CONFIG["use_channel_attn"] = True
        print(f"[v7 oam_overlap] 自适应覆盖: epochs={CONFIG['epochs']} mid_ch={CONFIG['mid_ch']} "
              f"num_layers={CONFIG['num_layers']} channel_attn={CONFIG['use_channel_attn']}", flush=True)

    # ---------- 1. 全局 RPP (跨 stage 共享) ----------
    os.makedirs("./data", exist_ok=True)
    rpp_system = generate_rpp(CONFIG["size"], device)

    # ---------- 2. Curriculum Learning 4 stage ----------
    if CONFIG.get("curriculum", True):
        stages = CONFIG["curriculum_stages"]
        n_stages = len(stages)
        threshold = CONFIG["curriculum_psnr_threshold"]
        print(f"\n[v7 CURRICULUM] {n_stages} stages, PSNR_C threshold={threshold} dB", flush=True)

        stage_results = []
        for s_idx, stage_cfg in enumerate(stages):
            best_psnr, best_state, best_sr = train_one_stage(
                stage_cfg=stage_cfg,
                stage_idx=s_idx,
                n_stages=n_stages,
                device=device,
                rpp_system=rpp_system,
                quick_test_n=CONFIG.get("quick_test_n", 200),
                num_layers=CONFIG["num_layers"],
                mid_ch=CONFIG["mid_ch"],
                use_channel_attn=CONFIG["use_channel_attn"],
                layout=CONFIG["layout"],
            )
            stage_results.append((s_idx, stage_cfg["n_channels"], best_psnr, best_sr))
            torch.save({
                'stage': s_idx + 1,
                'n_channels': stage_cfg["n_channels"],
                'l_auth': stage_cfg["l_auth"],
                'z_list': stage_cfg["z_list"],
                'model_state_dict': best_state,
                'psnr_center': best_psnr,
                'sec_ratio': best_sr,
            }, f"oam_crypt_v7_stage{s_idx+1}_best.pth")
            print(f"  [v7 STAGE {s_idx+1} DONE] n_ch={stage_cfg['n_channels']} "
                  f"PSNR_C={best_psnr:.2f} dB SR_OAM={best_sr:.4f} "
                  f"saved to oam_crypt_v7_stage{s_idx+1}_best.pth", flush=True)
            if s_idx < n_stages - 1 and best_psnr < threshold:
                print(f"  [v7 WARNING] Stage {s_idx+1} PSNR_C={best_psnr:.2f} < threshold {threshold}, "
                      f"continuing to next stage (curriculum strategy)", flush=True)

        # 加载最后 stage 最佳模型作为最终 v7 模型
        final_ckpt = torch.load(f"oam_crypt_v7_stage{n_stages}_best.pth", map_location=device)
        final_model = OAM_Crypt_D2NN(
            size=CONFIG["size"], num_layers=CONFIG["num_layers"],
            wavelength=CONFIG["wavelength"], pixel_size=CONFIG["pixel_size"],
            z_layer=CONFIG["z_layer"], z0=CONFIG["z0"], rpp=rpp_system,
            oam_keys=stages[-1]["l_auth"], z_list=stages[-1]["z_list"],
            obj_encoding=CONFIG["obj_encoding"],
            theta_max=np.deg2rad(CONFIG["theta_max_deg"]) if CONFIG.get("theta_max_deg") else None,
            slm_aware=CONFIG["slm_aware"],
            use_channel_attn=CONFIG["use_channel_attn"],
            mid_ch=CONFIG["mid_ch"],
            iterative_refine=CONFIG.get("iterative_refine", True),
            n_passes=CONFIG.get("n_passes", 3),
            oam_freq_filter=CONFIG.get("oam_freq_filter", True),
        ).to(device)
        final_model.load_state_dict(final_ckpt['model_state_dict'])
        final_model.eval()

        torch.save({
            'version': 'v7',
            'model_state_dict': final_model.state_dict(),
            'stage_results': stage_results,
            'psnr_center': final_ckpt['psnr_center'],
            'sec_ratio': final_ckpt['sec_ratio'],
        }, "oam_crypt_v7_final.pth")
        print(f"\n[v7 FINAL] PSNR_C={final_ckpt['psnr_center']:.2f} dB "
              f"SR_OAM={final_ckpt['sec_ratio']:.4f} saved to oam_crypt_v7_final.pth", flush=True)
    else:
        # 兼容 v6: 单 stage 全 10 通道
        print("[v7 WARNING] curriculum=False, 退化到 v6 训练模式", flush=True)
        stages = [{"n_channels": len(CONFIG["l_auth"]), "l_auth": CONFIG["l_auth"],
                   "epochs": CONFIG["epochs"], "lr": CONFIG["lr"], "z_list": CONFIG["z_list"]}]
        train_one_stage(stage_cfg=stages[0], stage_idx=0, n_stages=1, device=device,
                        rpp_system=rpp_system, quick_test_n=CONFIG.get("quick_test_n", 200),
                        num_layers=CONFIG["num_layers"], mid_ch=CONFIG["mid_ch"],
                        use_channel_attn=CONFIG["use_channel_attn"], layout=CONFIG["layout"])

    print("\n[v7] 训练流程结束")

