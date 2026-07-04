# OAM-Crypt-D2NN: 双密钥全光多用户信息加密/解密与隐写衍射神经网络

一套完整的 PyTorch 数值仿真实现:基于 OAM(轨道角动量)螺旋相位 + RPP(随机相位板)双密钥体系,通过衍射神经网络(D2NN)实现多用户全光解密,并兼容纯相位 SLM 加载。

## 物理流程

### 加密
```
明文 P_i -> 物光相位化 exp(i·π·P_i) -> ASM(z0) 传播 -> ×OAM_i 螺旋相位 -> 4 路叠加 -> ×RPP
                                                                                              |
                                                                                         密文 U_cipher
```

### 解密
```
U_cipher -> [纯相位 SLM 加载: exp(i·arg(U))] -> ×conj(RPP) -> ×conj(OAM_j) [4 路] -> ASM(-z0) -> D2NN -> U-Net
```

## 关键设计

| 特性 | 实现 |
|------|------|
| **双密钥** | OAM 拓扑荷 (用户级寻址) + RPP 随机相位 (系统级加密) |
| **多用户** | 4 个授权 OAM 密钥 (l=±1, ±3), 每张图像独立通道 |
| **空间分离** | 4 张明文放在 4 个象限, 降低 OAM 串扰 |
| **物光相位化** | `exp(i·π·P)` 让 \|U\|≡1, 信息编码到相位, SLM 纯相位加载无损 |
| **训练=部署** | forward 开头相位化, 消除 train-test mismatch |
| **12 通道 U-Net** | 4 real + 4 imag + 4 phase, 多视角融合精修 |
| **AMP 混合精度** | 支持 1080×1080 大尺寸训练 (~25 min/epoch on RTX 3090) |

## 性能 (1080×1080, 532nm)

| 指标 | 数值 |
|------|------|
| 分辨率 | 1080×1080 |
| 波长 | 532 nm (绿光) |
| 训练轮次 | 20 epoch (warmup) |
| **纯相位 SLM 解密 PSNR** | **38.02 dB** |
| SecurityRatio | 0.0003 (越低越安全) |

## 文件结构

```
.
├── oam_crypt_d2nn.py            # 主训练代码 (CONFIG, 加密, 解密网络, 训练循环)
├── generate_slm_phase.py        # 生成 1080×1080 纯相位 SLM 加载图
├── test_slm_schemes.py           # SLM 加载方案对比测试
├── eval_checkpoint.py            # 快速评估 checkpoint PSNR
├── oam_crypt_dnn_epoch_20.pth   # 训练好的解密网络 (1080 版本, PSNR 38.02 dB)
├── rpp_system.pt                 # 系统密钥 RPP
├── slm_output_1080/              # SLM 加载图输出
│   ├── slm_phase_1080x1080.png   # 仿真用相位图
│   ├── slm_phase_1920x1080.png   # Holoeye PLUTO SLM 加载图
│   └── slm_overview.png          # 可视化概览
├── eval_plot.png                 # 解密结果可视化
├── final_security_plot.png       # 安全性对比图
└── slm_scheme_comparison.png     # 三方案对比图
```

## 快速开始

### 1. 训练
```bash
py oam_crypt_d2nn.py
```

### 2. 生成 SLM 加载图
```bash
py generate_slm_phase.py oam_crypt_dnn_epoch_20.pth
```

### 3. 评估 checkpoint
```bash
py eval_checkpoint.py oam_crypt_dnn_epoch_20.pth
```

## SLM 加载说明

适用于 **Holoeye PLUTO-2.1** (1920×1080, 8.0 μm, 纯相位):

1. 将 `slm_output_1080/slm_phase_1920x1080.png` 加载到 SLM 控制软件
2. SLM 工作波长设为 **532 nm**
3. SLM 出射光场 = `exp(i·arg(U_cipher))` (纯相位, 振幅恒为 1)
4. 1080×1080 全息图水平居中放置 (x=[420, 1500], y=[0, 1080])
5. 后续接入解密光路 (RPP 去除 → OAM 解复用 → 传播 → D2NN/U-Net)

## 依赖

```
torch, torchvision, numpy, matplotlib, Pillow
```

## 物理参数

| 参数 | 数值 |
|------|------|
| 系统尺寸 | 1080×1080 |
| 波长 | 532 nm |
| 像素尺寸 | 8.0 μm |
| 传播距离 z0 | 0.1 m |
| OAM 密钥 | l ∈ {-3, -1, +1, +3} |
| 错误 OAM | l ∈ {-2, 0, +2, +4} |

## 许可

MIT License
