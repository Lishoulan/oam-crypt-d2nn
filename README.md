# OAM Cryptographic Diffractive Neural Network (OAM Crypt-D2NN)

基于**轨道角动量(OAM)** 的多用户图像加密/解密衍射神经网络系统,使用纯相位 SLM 实现。

## 阶段性成果(Milestone v1.0)

### 核心指标

| 维度 | 数字仿真 | SLM 仿真 (8-bit 加载) | 损耗 |
|------|---------|---------------------|------|
| **中心 270×270 PSNR** | **22.07 dB** | **21.04 dB** | **1.03 dB** ✓ |
| 4 通道平均 | 22.07 | 21.04 | < 1.5 dB |

### 关键技术成果

- **OAM 多通道复用**: 4 个 OAM 通道 (`l=±15, ±5`) 复用同一光斑,通过多平面 (z=0.10/0.18/0.26/0.34 m) 区分
- **SLM 感知训练**: 在模型 forward 内部模拟 SLM 8-bit 相位量化,训练时学到的就是 SLM 加载后的真实分布
- **SLM 加载损耗修复**: 从 12.71 dB 降至 1.03 dB(降幅 92%)
- **DPE + K 空间约束**: 双相位编码适配纯相位 SLM; K 空间约束 `θ_max=1.5°` 让训练相位更平滑
- **U-Net 精修层**: 中心加权 MSE + L1 损失,跨层融合去除 OAM 解调残余噪声

## 物理架构

```
明文图像(4 张 270×270) 
   ↓ exp(iπP) 相位编码
   ↓ 4 路 OAM 调制(l=±15, ±5)
   ↓ 多平面 ASM 聚焦
   ↓ × RPP 随机相位密钥
   ↓
U_cipher (1080×1080 复振幅)  ← 密文 cipher
   ↓ DPE + 8-bit 灰度
   ↓ SLM 加载 + 衍射
   ↓
棋盘格 phase only 场
   ↓ 数字解密: 去除 RPP + 4 路 OAM 解调
   ↓ ASM 反向传播
   ↓ 2 层 D2NN 衍射
   ↓ U-Net 精修
   ↓
4 路重建图像
```

## 文件清单

### 核心代码

- `oam_crypt_d2nn.py` — 主训练脚本(模型定义 + 训练循环)
- `slm_load_test.py` — SLM 加载验证(数字 vs SLM 8-bit 仿真对比)
- `attack_oam_test.py` — OAM 攻击测试(错误密钥响应)
- `eval_checkpoint.py` — 模型评估
- `test_slm_schemes.py` — SLM 方案对比(透射 vs 反射)
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

- `slm_loading_test.png` — SLM 加载测试 (4 通道 数字 vs SLM 对比)
- `final_security_plot.png` — 安全性曲线
- `multi_plane_quick_verify.png` — 多平面聚焦验证
- `slm_scheme_comparison.png` — SLM 方案对比
- `attack_oam_heatmap.png` / `attack_oam_images.png` — OAM 攻击分析
- `eval_plot.png` / `results.png` — 评估结果
- `slm_output/` — SLM 8-bit 全息图(可加载到 Holoeye PLUTO)

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 训练

```bash
python oam_crypt_d2nn.py  # 8 epoch SLM 感知训练 (~22 min on RTX 3090)
```

### SLM 加载验证

```bash
python slm_load_test.py  # 生成 slm_loading_test.png
```

## 配置

主配置在 `oam_crypt_d2nn.py` 的 `CONFIG` 字典:

| 参数 | 值 | 说明 |
|------|----|----|
| `l_auth` | `[-15, -5, 5, 15]` | 4 个 OAM 通道 |
| `z_list` | `[0.10, 0.18, 0.26, 0.34]` | 4 个解码平面 |
| `num_layers` | `2` | D2NN 衍射层数 |
| `theta_max_deg` | `1.5` | K 空间约束最大传播角 |
| `slm_aware` | `True` | SLM 8-bit 量化感知训练 |
| `size` | `1080` | SLM 网格尺寸 |
| `wavelength` | `532e-9` | 绿光波长 |
| `pixel_size` | `8e-6` | Holoeye PLUTO 像素 |
| `epochs` | `8` | 训练轮数 |
| `obj_encoding` | `phase` | 相位编码 |

## 技术栈

- **PyTorch** + CUDA(AMP 混合精度)
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
