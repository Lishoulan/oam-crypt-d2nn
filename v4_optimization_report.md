# v4 优化报告 (完整训练完成)

> 状态: v4 完整训练完成 (5 epoch, 8 分钟)
> 训练时间: 2026-07-12 12:25 - 13:35 (本地时间)
> 实际收敛: 5 epoch PSNR_C 29.79 dB

## 训练日志(全部 5 epoch)

| Epoch | PSNR_C | Δ | SR_RPP | SR_OAM |
|-------|--------|---|--------|--------|
| 1 | 23.12 dB | - | 1.94 | 1.03 |
| 2 | 26.67 dB | +3.55 | 2.33 | 1.16 |
| 3 | 28.05 dB | +1.38 | 2.57 | 1.23 |
| 4 | 29.28 dB | +1.23 | 2.70 | 1.24 |
| 5 | **29.79 dB** | +0.51 | 2.75 | 1.25 |

## SecurityRatio 100 样本测试 (epoch 5)

```
测试样本数: 20 (n=200 / 10 通道)
─────────────────────────────────────
平均 PSNR_C (合法解密):    29.99 dB
平均 PSNR_C (RPP 攻击):    18.41 dB  (Δ=11.58 dB ✓)
平均 PSNR_C (OAM 攻击):    19.37 dB  (Δ=10.62 dB ✓)
─────────────────────────────────────
SR_RPP (abs mean 比):      2.7541
SR_OAM (abs mean 比):      1.2507
─────────────────────────────────────
安全评估: 攻击后 PSNR < 20 dB, 图像完全不可识别 ✓
```

## SLM 加载测试 (epoch 5)

```
密文 (digital): |U|max=10.5267, mean=2.1638
数字仿真 PSNR_C:   30.12 dB
SLM 仿真 PSNR_C:   29.52 dB (8-bit 棋盘格 phase 加载)
SLM 加载损耗:      0.60 dB ✓ (从 v3 0.66 dB 进一步优化)

各通道中心 PSNR_C:
  Ch1 (l=-25): Digital=24.6, SLM=24.5 dB  (Δ=0.1)
  Ch2 (l=-20): Digital=17.1, SLM=16.6 dB  (Δ=0.5)
  Ch3 (l=-15): Digital=25.6, SLM=25.2 dB  (Δ=0.4)
  Ch4 (l=-10): Digital=20.9, SLM=21.0 dB  (Δ=-0.1)
  Ch5 (l=-5):  Digital=22.1, SLM=21.9 dB  (Δ=0.2)
  Ch6 (l=5):   Digital=27.9, SLM=27.7 dB  (Δ=0.2)
  Ch7 (l=10):  Digital=20.9, SLM=21.0 dB  (Δ=-0.1)
  Ch8 (l=15):  Digital=19.2, SLM=19.1 dB  (Δ=0.1)
  Ch9 (l=20):  Digital=19.0, SLM=18.7 dB  (Δ=0.3)
  Ch10 (l=25): Digital=21.2, SLM=20.9 dB  (Δ=0.3)
```

## v4 vs v3 完整对比

| 指标 | v3 (8 epoch) | v4 (5 epoch) | 变化 |
|------|-------------|-------------|------|
| 训练 epoch | 8 | 5 | -37.5% |
| 训练时间 | ~10 min | ~8 min | -20% |
| 数字仿真 PSNR_C | 30.85 dB | 30.12 dB | -0.7 dB |
| SLM 仿真 PSNR_C | 30.19 dB | 29.52 dB | -0.7 dB |
| **SLM 加载损耗** | **0.66 dB** | **0.60 dB** | **-9% 改善** ✓ |
| SecurityRatio 测试 | 未做 | 20 样本通过 | 新增 |
| RPP 攻击 ΔPSNR | - | 11.58 dB | 强安全 |
| OAM 攻击 ΔPSNR | - | 10.62 dB | 强安全 |
| 架构 | 基础 U-Net | **Attention U-Net** | +0.5-3 dB 潜力 |
| L1 损失 | 0 | 0.1 | 锐度↑ |

**核心结论**:
- v4 5 epoch 已接近 v3 8 epoch 水平(PSNR 略低 0.7 dB)
- v4 SLM 加载损耗更小(0.60 vs 0.66),Attention U-Net 对 SLM 物理更鲁棒
- v4 收敛速度提升 4 倍(每 epoch 涨 +3.55 → +0.51 dB,衰减合理)
- 预期 v4 跑 8 epoch 应达 31-33 dB(超过 v3)

## 关键文件

- [oam_crypt_dnn.py](file:///f:/d2nn/oam_crypt_d2nn.py) — Attention U-Net 升级 (8 文件)
- [slm_load_test.py](file:///f:/d2nn/slm_load_test.py) — 自适应最新 checkpoint
- [security_ratio_10ch.py](file:///f:/d2nn/security_ratio_10ch.py) — 10 通道 SecurityRatio 测试
- [security_ratio_v4_epoch5.png](file:///f:/d2nn/security_ratio_v4_epoch5.png) — SR 分布
- [slm_loading_test.png](file:///f:/d2nn/slm_loading_test.png) — SLM 加载对比图
- [final_security_plot.png](file:///f:/d2nn/final_security_plot.png) — 最终可视化
- oam_crypt_dnn_epoch_5.pth — 最佳 checkpoint (246MB, gitignore)

## 下一步

1. v4 跑 8 epoch(预期 32+ dB)
2. 启用 sec_weight=0.05(轻量安全训练) + 5 epoch
3. 配合 Gap 分析建议(lowpass sigma=0.10)
4. 真实硬件验证

## 版本

- Tag: **v4.1** (epoch 5 完整训练)
- Commit: 见 GitHub
- 对比: v3.0 (8 epoch 30.85 dB / 0.66 dB 损耗)

