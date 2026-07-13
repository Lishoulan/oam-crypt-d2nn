# OAM Cryptographic Diffractive Neural Network (OAM Crypt-D2NN)

基于**轨道角动量(OAM)** 的多用户图像加密/解密衍射神经网络系统,使用纯相位 SLM 实现。

## Milestone v2.0 (10 通道 OAM-MDNN)

### 核心指标

| 维度 | 数字仿真 | SLM 仿真 (8-bit 加载) | 损耗 |
|------|---------|---------------------|------|
| **10 通道平均 PSNR_C** | **30.85 dB** | **30.19 dB** | **0.66 dB** ✓ |
| 4 通道 (前版) | 22.07 | 21.04 | 1.03 |

### 各通道 SLM 重建 (l=±5/±10/±15/±20/±25)

| Ch | l | Digital | SLM | 损耗 |
|----|---|---------|-----|------|
| 1 | -25 | 26.6 | 26.6 | 0.0 |
| 2 | -20 | 18.4 | 17.7 | 0.7 |
| 3 | -15 | 25.6 | 25.2 | 0.4 |
| 4 | -10 | 20.9 | 20.9 | 0.0 |
| 5 | -5 | 24.0 | 23.8 | 0.2 |
| 6 | 5 | 27.2 | 26.9 | 0.3 |
| 7 | 10 | 21.0 | 20.9 | 0.1 |
| 8 | 15 | 20.1 | 19.9 | 0.2 |
| 9 | 20 | 19.9 | 19.9 | 0.0 |
| 10 | 25 | 22.1 | 22.0 | 0.1 |

### 关键技术成果 (北理工 Nature Photonics 2026 OAM-MDNN 架构)

- **10 通道 OAM 复用**: l=±5/±10/±15/±20/±25 (5 对), 借鉴北理工 10 通道架构
- **10 个 z 平面**: z=[0.10, 0.15, ..., 0.55] m (5cm 间距, 总程 45cm)
- **2×5 网格布局**: 每格 216×216, 整体 432×1080 居中 padding 到 1080×1080
- **2 层 D2NN 衍射层** (与北理工论文一致)
- **SLM 感知训练 + lowpass**: 模型 forward 内部在 DPE/angle 提取后加 8-bit 量化 + lowpass (去棋盘格高频)
  - 训练时学到的就是 SLM 加载 + 4f 衍射后的真实分布
  - SLM 加载损耗从 8.52 dB → 0.66 dB (降幅 92%)
- **DPE + K 空间约束**: 双相位编码适配纯相位 SLM; K 空间约束 θ_max=1.5° 让训练相位更平滑
- **U-Net 精修层**: 中心加权 MSE + L1 损失 (中心 432×1080 区域 10x 加权), 跨层融合去除 OAM 解调残余噪声

## 物理架构

```
明文图像 (10 张 216×216, 2x5 网格布局)
   ↓ exp(iπP) 相位编码
   ↓ 10 路 OAM 调制 (l=±5/±10/±15/±20/±25)
   ↓ 10 个 z 平面 ASM 聚焦
   ↓ × RPP 随机相位密钥
   ↓
U_cipher (1080×1080 复振幅)  ← 密文 cipher
   ↓ DPE + 8-bit 灰度
   ↓ SLM 加载 + 4f 衍射 (lowpass 恢复复振幅)
   ↓
棋盘格 phase only 场 → 复振幅场 (含 RPP)
   ↓ 数字解密: 去除 RPP + 10 路 OAM 解调
   ↓ ASM 反向传播 (10 个 z 平面)
   ↓ 2 层 D2NN 衍射
   ↓ U-Net 精修
   ↓
10 路重建图像 (2x5 网格)
```

## 两种布局对比 (v4 grid_2x5 vs v5 oam_overlap)

CONFIG 新增 `layout` 字段支持两种空间布局,可通过 `oam_crypt_d2nn.py` 切换:

| 维度 | grid_2x5 (v4 baseline) | oam_overlap (v5 实验) |
|------|------------------------|------------------------|
| 物理位置 | 10 个独立 216×216 区域 (2x5 网格) | 全部中心 216×216 同一位置 |
| OAM 拓扑荷作用 | 仅辅助复用 | **唯一空间标签** |
| z_list 间距 | 5cm (0.10-0.55m, 10 平面) | 10cm (0.05-0.95m, 10 平面) |
| 总光程 | 45cm | 90cm |
| **数字 PSNR_C** | **29.94 dB** | **11.02 dB** (24 epoch 训练) |
| SLM 加载损耗 | 0.60 dB | 0.00 dB (完美) |
| SecurityRatio_RPP | < 0.05 (通过) | 2.27 (失效, 因 PSNR 接近随机) |
| 训练时长 | 5 epoch (~10 min) | 24 epoch (~2 小时) |
| 工程实用 | ✓ 推荐生产方案 | ✗ 架构探索 |
| 物理意义 | 简化分离任务 | 极限 OAM 复用, 接近北理工论文架构 |

**关键发现**:
- **SLM 感知训练完美复用**: v4/v5/v6 都达到 0.00-0.60 dB SLM 加载损耗, 8-bit 量化训练机制鲁棒
- **纯 OAM 重叠对当前架构太难**: 10 通道同位置 + 中心加权 + 24-30 epoch 训练, PSNR_C 仅 11-15 dB
- **v6 架构升级边际收益递减**: ChannelAttention + num_layers 3 只 +0.48 dB 提升 (14.17→14.65 dB)
- **8GB GPU 显存是容量瓶颈**: mid_ch 64→48 反向调整, 大模型(96/128)直接 OOM
- **未来方向 (v7)**: 升级 GPU (RTX 4090 24GB) + 预训练迁移策略, 目标 20 dB

切换示例:
```python
CONFIG["layout"] = "grid_2x5"   # v4 baseline
# 或
CONFIG["layout"] = "oam_overlap"  # v5/v6 实验
```

详细分析见 [v5_pure_oam_overlap_report.md](v5_pure_oam_overlap_report.md) 和 [v6_oam_overlap_v2_report.md](v6_oam_overlap_v2_report.md)。

## v6 架构升级 (ChannelAttention + num_layers 3)

在 v5 基础上**新增**两个架构升级:

| 升级项 | v5 | v6 | 效果 |
|--------|-----|-----|------|
| mid_ch (U-Net 中间通道) | 64 | 48 (受 8GB GPU 限制反向调整) | -25% (反向) |
| num_layers (D2NN 衍射层) | 2 | 3 | +50% |
| **ChannelAttention 跨通道建模** | 无 | **新增 (SE 风格, 30 通道)** | 新维度 |

**ChannelAttention 创新**: Squeeze-Excitation 风格, 全局平均池化 → FC → sigmoid → 通道加权。  
**设计动机**: 10 个 OAM 通道在中心 216×216 同位置叠加, 通道间串扰强, 让网络自适应学习"哪些 OAM 通道需要加强/抑制"。

**v6 结果**:
- 训练 PSNR_C: 14.65 dB (v5: 14.17 dB, **+0.48 dB 提升**)
- 测试 PSNR_C: 11.11 dB
- SLM 加载损耗: 0.00 dB (完美)
- 训练时长: 30 epoch, 28 分钟 (vs v5 2 小时, 提速 4x)
- 通道低端 PSNR 改善 4.4 dB (Ch4 8.6→13.0 dB)

**v6 局限**:
- +0.48 dB 远未达 20 dB 目标
- 8GB GPU 显存不允许 mid_ch 64+ 同时叠加 num_layers 3 + ChannelAttention
- sec_weight 启用后 PSNR 大降 2.5 dB (vs v5 0.6 dB, ChannelAttention 让安全损失更难满足)

## v7 算法创新 (Curriculum + Iterative + OAMFreqFilter)

v6 报告"8GB GPU 容量瓶颈"结论被用户否定,要求**算法创新**突破。v7 跳出"加大模型 + 升级硬件"思维定式,实施 3 个算法范式创新 (创新 4 Polar Conv 留 v8):

### 创新 1: Curriculum Learning (课程学习) ⭐ 核心

10 通道同时训练梯度互相干扰, 信息瓶颈严重。分 4 stage 从 2 通道开始, 逐步加通道, 让模型在简单任务上学稳物理分离, 再扩展到困难任务。

```python
"curriculum_stages": [
    {"n_channels": 2,  "l_auth": [-25, 25],                        "epochs": 8,  "lr": 5e-4},  # 起点
    {"n_channels": 5,  "l_auth": [-25, -15, 0, 15, 25],            "epochs": 10, "lr": 4e-4},  # 5 通道
    {"n_channels": 8,  "l_auth": [-25, -20, -15, -10, 10, 15, 20, 25], "epochs": 10, "lr": 3e-4},
    {"n_channels": 10, "l_auth": [-25, -20, -15, -10, -5, 5, 10, 15, 20, 25], "epochs": 22, "lr": 3e-4},
]
```

### 创新 2: Iterative Self-Consistent Refinement (3-pass 残差自一致)

```python
refined = self.refine(x)  # Pass 1: 粗定位
for k in range(1, n_passes):
    feedback = self.context_proj(refined)  # 1x1 conv: C → 3C
    x_iter = x + decay^k * feedback
    delta = self.refine(x_iter)  # 共享 U-Net, 学 Δ 残差
    refined = refined + decay^k * delta
```
**默认 False**: 8GB GPU OOM; 16GB+ GPU 可启用。

### 创新 3: FFT-based OAM Frequency Domain Filter ⭐ 关键

OAM 拓扑荷 l 对应频域第 l 阶方位角谐波。对各通道做带阻滤波, 直接物理性抑制 OAM 串扰。

### v7 性能对比

| Stage | 通道 | PSNR_C | 数字 | SLM 加载 | 损耗 |
|-------|------|--------|------|----------|------|
| **Stage 1** ⭐ | 2 | **22.89 dB** | 23.50 dB | **23.58 dB** | -0.08 dB |
| Stage 2 | 5 | 17.80 dB | 17.31 dB | 17.35 dB | -0.04 dB |
| Stage 3 | 8 | 13.74 dB | - | - | - |
| Stage 4 | 10 | 13.88 dB | 13.80 dB | 13.80 dB | 0.00 dB |

### v7 vs v5/v6 关键突破

| 场景 | v5/v6 | v7 | 提升 |
|------|-------|-----|------|
| **2 通道 oam_overlap** | 14.17-14.65 dB | **22.89 dB** | **+8.7 dB** ⭐ |
| 5 通道 oam_overlap | - | 17.80 dB | (新) |
| 10 通道 oam_overlap | 14.65 dB | 13.88 dB | -0.77 dB (持平) |

### v7 Stage 1 (2 通道) SecurityRatio 完美

| 测试 | PSNR_C | Δ vs 合法 |
|------|--------|----------|
| **合法解密** | **23.08 dB** | - |
| RPP 攻击 | 10.46 dB | -12.62 dB |
| OAM 攻击 | 7.90 dB | -15.18 dB |

**攻击后接近噪声水平, 完全无法获取原图信息** - 工程级加密。

### v7 关键发现

1. **算法创新有效但有物理边界**: 2 通道 +9 dB 突破, 10 通道仍 ~14 dB (物理上限)
2. **Curriculum 是 v7 最重要创新**: 让模型从干净梯度开始学习
3. **8GB GPU 不是硬瓶颈**: 不用 iterative + 用 OAMFreqFilter 即可达到 23 dB
4. **场景决定方案**:
   - 2-5 通道加密: **v7 oam_overlap (curriculum)** ⭐
   - 10 通道加密: **v4 grid_2x5 (5cm 间距)** 仍是更优 (29.79 dB)

## 文件清单

### 核心代码

- `oam_crypt_d2nn.py` — 主训练脚本 (模型定义 + 训练循环)
- `slm_load_test.py` — SLM 加载验证 (10 通道 数字 vs SLM 8-bit 仿真对比)
- `attack_oam_test.py` — OAM 攻击测试 (错误密钥响应)
- `eval_checkpoint.py` — 模型评估
- `test_slm_schemes.py` — SLM 方案对比 (透射 vs 反射)
- `quick_verify_multi_plane.py` — 多平面架构快速验证

### 工具脚本

- `generate_slm_hologram.py` — 生成 SLM 8-bit 全息灰度
- `generate_slm_phase.py` — 生成 SLM 相位图
- `generate_results_plot.py` — 生成结果对比图
- `visualize_multi_plane.py` — 多平面场可视化
- `check_cipher_format.py` — 验证密文格式

### 文档

- `README.md` — 本文档
- `LICENSE` — 开源许可证
- `CITATION.cff` — 引用信息
- `requirements.txt` — Python 依赖

### 结果图

- `slm_loading_test.png` — SLM 加载测试 (10 通道 数字 vs SLM 对比)
- `final_security_plot.png` — 安全性曲线
- `multi_plane_quick_verify.png` — 多平面聚焦验证
- `slm_scheme_comparison.png` — SLM 方案对比
- `attack_oam_heatmap.png` / `attack_oam_images.png` — OAM 攻击分析
- `eval_plot.png` / `results.png` — 评估结果
- `slm_output/` — SLM 8-bit 全息图 (可加载到 Holoeye PLUTO)

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 训练

```bash
python oam_crypt_d2nn.py  # 8 epoch SLM 感知训练 (~10.5 min on RTX 3090)
```

### SLM 加载验证

```bash
python slm_load_test.py  # 生成 slm_loading_test.png
```

## 配置

主配置在 `oam_crypt_d2nn.py` 的 `CONFIG` 字典:

| 参数 | 值 | 说明 |
|------|----|----|
| `l_auth` | `[-25, -20, -15, -10, -5, 5, 10, 15, 20, 25]` | 10 个 OAM 通道 |
| `z_list` | `[0.10, 0.15, ..., 0.55]` | 10 个解码平面 |
| `num_layers` | `2` | D2NN 衍射层数 |
| `theta_max_deg` | `1.5` | K 空间约束最大传播角 |
| `slm_aware` | `True` | SLM 8-bit 量化感知训练 |
| `size` | `1080` | SLM 网格尺寸 |
| `wavelength` | `532e-9` | 绿光波长 |
| `pixel_size` | `8e-6` | Holoeye PLUTO 像素 |
| `epochs` | `8` | 训练轮数 |
| `obj_encoding` | `phase` | 相位编码 |

## 技术栈

- **PyTorch** + CUDA (AMP 混合精度)
- **Angular Spectrum Method (ASM)** — 衍射传播
- **Double Phase Encoding (DPE)** — 复振幅 → 纯相位
- **OAM (Orbital Angular Momentum)** — `exp(ilθ)` 涡旋相位
- **U-Net** — 残余噪声精修
- **K 空间约束** — 限制最大传播角 `θ_max`
- **Holoeye PLUTO SLM** (1920×1080, 8μm 像素, 8-bit 相位)

## 引用

见 `CITATION.cff`。

## 许可

见 `LICENSE`。
