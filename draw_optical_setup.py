# -*- coding: utf-8 -*-
"""
光路图绘制脚本
==============
用 matplotlib 绘制 4f + SLM + 相机的实验室光路图(俯视图 + 3D 立体视图)

输出:
  - optical_setup_2d.png     俯视图(机械设计参考)
  - optical_setup_3d.png     3D 立体图(原理演示)
  - optical_setup_principle.png  原理示意图(频谱面 + pinhole)
"""

import numpy as np
import matplotlib.pyplot as plt
from font_config import setup_cjk
setup_cjk()
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle, FancyBboxPatch
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches


# ==================== 配置参数 ====================

# 项目物理参数
SLM_PIXEL = 8e-6       # m
WAVELENGTH = 532e-9    # m
F1 = 200e-3            # m (4f 第一片)
F2 = 200e-3            # m (4f 第二片)
SLM_SIZE_X = 1920 * SLM_PIXEL  # 15.36 mm
SLM_SIZE_Y = 1080 * SLM_PIXEL  # 8.64 mm

# 光路尺寸(布局)
L_SF_L1 = 50e-3        # 空间滤波器 → 准直镜
L_L1_SLM = 200e-3      # 准直镜 → SLM
L_SLM_L2 = F1          # SLM → L2
L_L2_FREQ = F1         # L2 → 频谱面
L_FREQ_L3 = F2         # 频谱面 → L3
L_L3_CAM = F2          # L3 → 相机


# ==================== 俯视图 2D ====================

def draw_2d(save_path="optical_setup_2d.png"):
    """画 2D 俯视图(标注距离)"""
    fig, ax = plt.subplots(figsize=(18, 6))

    # 水平线(光轴)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)

    # 设备 x 坐标(累积)
    x_laser = 0
    x_sf = 0.05
    x_l1 = x_sf + L_SF_L1
    x_slm = x_l1 + L_L1_SLM
    x_l2 = x_slm + L_SLM_L2
    x_freq = x_l2 + L_L2_FREQ
    x_l3 = x_freq + L_FREQ_L3
    x_cam = x_l3 + L_L3_CAM

    # 1. 激光器
    ax.add_patch(FancyBboxPatch((x_laser-0.02, -0.04), 0.04, 0.08,
                                 boxstyle="round,pad=0.005",
                                 facecolor='#FFE4B5', edgecolor='black', linewidth=1.5))
    ax.text(x_laser, -0.10, "532 nm\n激光器", ha='center', va='top', fontsize=10, weight='bold')

    # 2. 空间滤波器
    ax.add_patch(Circle((x_sf, 0), 0.025, facecolor='#E0E0E0', edgecolor='black', linewidth=1.5))
    ax.text(x_sf, -0.10, "空间滤波器\n(20× + 5μm pinhole)", ha='center', va='top', fontsize=9)

    # 3. 准直透镜 L1
    ax.add_patch(mpatches.Ellipse((x_l1, 0), 0.04, 0.16, facecolor='#87CEEB', edgecolor='black', linewidth=1.5))
    ax.text(x_l1, -0.10, f"准直镜 L1\nf={F1*1e3:.0f}mm", ha='center', va='top', fontsize=9)

    # 4. SLM
    slm_height = 0.10
    ax.add_patch(Rectangle((x_slm-0.02, -slm_height/2), 0.04, slm_height,
                            facecolor='#FFB6C1', edgecolor='black', linewidth=2))
    ax.text(x_slm, -0.13, f"Holoeye PLUTO\n{SLM_SIZE_X*1e3:.1f}×{SLM_SIZE_Y*1e3:.1f}mm\n1920×1080",
            ha='center', va='top', fontsize=9, weight='bold')
    # SLM 倾角(反射)
    ax.annotate("", xy=(x_slm+0.05, 0.10), xytext=(x_slm+0.02, 0.0),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax.text(x_slm+0.05, 0.15, "反射光", fontsize=8, color='red')

    # 5. L2 透镜(在 SLM 反射后)
    x_l2 = x_slm + 0.10
    ax.add_patch(mpatches.Ellipse((x_l2, 0.10), 0.04, 0.16, facecolor='#87CEEB', edgecolor='black', linewidth=1.5))
    ax.text(x_l2, 0.18, f"L2\nf={F1*1e3:.0f}mm", ha='center', va='bottom', fontsize=9)
    ax.plot([x_slm+0.04, x_l2-0.02], [0.0, 0.10], 'r-', linewidth=1.5, alpha=0.7)

    # 6. 频谱面(pinhole)
    x_freq = x_l2 + 0.20
    ax.add_patch(Circle((x_freq, 0.10), 0.012, facecolor='gold', edgecolor='black', linewidth=2))
    ax.text(x_freq, 0.18, "Pinhole\n0.1-0.2mm", ha='center', va='bottom', fontsize=9, weight='bold')

    # 7. L3 透镜
    x_l3 = x_freq + 0.20
    ax.add_patch(mpatches.Ellipse((x_l3, 0.10), 0.04, 0.16, facecolor='#87CEEB', edgecolor='black', linewidth=1.5))
    ax.text(x_l3, 0.18, f"L3\nf={F2*1e3:.0f}mm", ha='center', va='bottom', fontsize=9)

    # 8. sCMOS 相机
    x_cam = x_l3 + 0.20
    ax.add_patch(FancyBboxPatch((x_cam-0.02, 0.05), 0.04, 0.10,
                                 boxstyle="round,pad=0.005",
                                 facecolor='#90EE90', edgecolor='black', linewidth=1.5))
    ax.text(x_cam, 0.18, "sCMOS\n≥2048²", ha='center', va='bottom', fontsize=9, weight='bold')

    # 9. 距离标注
    for x_pair, label in [
        (x_laser, "激光"),
        (x_sf, "扩束"),
        (x_l1, "准直"),
        (x_slm, "SLM"),
        (x_l2, "L2"),
        (x_freq, "频谱"),
        (x_l3, "L3"),
        (x_cam, "相机"),
    ]:
        ax.axvline(x=x_pair, ymin=0.4, ymax=0.5, color='gray', linestyle=':', alpha=0.5)

    # 10. 关键距离标注
    ax.annotate("", xy=(x_freq, 0.30), xytext=(x_l2, 0.30),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=1.2))
    ax.text((x_l2+x_freq)/2, 0.31, f"f1={F1*1e3:.0f}mm", ha='center', fontsize=9, color='blue')

    ax.annotate("", xy=(x_l3, 0.30), xytext=(x_freq, 0.30),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=1.2))
    ax.text((x_freq+x_l3)/2, 0.31, f"f2={F2*1e3:.0f}mm", ha='center', fontsize=9, color='blue')

    ax.annotate("", xy=(x_cam, 0.30), xytext=(x_l3, 0.30),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=1.2))
    ax.text((x_l3+x_cam)/2, 0.31, f"f2={F2*1e3:.0f}mm", ha='center', fontsize=9, color='blue')

    # 11. 标题
    ax.set_title("OAM-MDNN 实验室光路 (俯视图)\n"
                 f"SLM {SLM_SIZE_X*1e3:.1f}×{SLM_SIZE_Y*1e3:.1f}mm, 532 nm, 4f 系统 f1=f2=200mm",
                 fontsize=13, weight='bold')
    ax.set_xlim(-0.05, x_cam + 0.10)
    ax.set_ylim(-0.20, 0.40)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_path}")


# ==================== 3D 立体图(伪 3D) ====================

def draw_3d(save_path="optical_setup_3d.png"):
    """画伪 3D 立体图(用阴影和透视)"""
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 光路(沿 x 轴)
    x_positions = {
        'laser': 0,
        'sf': 0.5,
        'l1': 1.0,
        'slm': 2.0,
        'l2': 2.5,
        'freq': 3.5,
        'l3': 4.5,
        'cam': 5.5,
    }

    # 画光束(红色)
    z = np.linspace(-0.5, 0.5, 50)
    for key, x in x_positions.items():
        ax.plot([x, x], [0, 0], [-0.3, 0.3], 'r-', alpha=0.3, linewidth=0.5)

    # 设备
    # 激光
    ax.scatter([x_positions['laser']], [0], [0], s=300, c='orange', marker='s', label='532nm Laser')
    # 空间滤波器
    ax.scatter([x_positions['sf']], [0], [0], s=200, c='gray', marker='o', label='Spatial Filter')
    # 准直镜
    ax.scatter([x_positions['l1']], [0], [0], s=400, c='lightblue', marker='o', label='L1 Collimator')
    # SLM
    ax.scatter([x_positions['slm']], [0], [0], s=600, c='pink', marker='s', label='SLM (1920x1080)')
    # L2, L3
    ax.scatter([x_positions['l2']], [0], [0], s=400, c='lightblue', marker='o', label='L2 (4f-1)')
    ax.scatter([x_positions['l3']], [0], [0], s=400, c='lightblue', marker='o', label='L3 (4f-2)')
    # Pinhole
    ax.scatter([x_positions['freq']], [0], [0], s=300, c='gold', marker='*', label='Pinhole (Fourier)')
    # Camera
    ax.scatter([x_positions['cam']], [0], [0], s=400, c='lightgreen', marker='s', label='sCMOS Camera')

    # 标注
    for key, x in x_positions.items():
        label = {
            'laser': '532nm\nLaser',
            'sf': 'Spatial\nFilter',
            'l1': 'L1\nf=200mm',
            'slm': 'Holoeye PLUTO\n1920×1080',
            'l2': 'L2\nf=200mm',
            'freq': 'Pinhole\n0.1-0.2mm',
            'l3': 'L3\nf=200mm',
            'cam': 'sCMOS\n≥2048²',
        }[key]
        ax.text(x, 0, -0.5, label, ha='center', va='top', fontsize=9, weight='bold')

    # 距离标注(蓝色箭头)
    for x_start, x_end, label in [
        (x_positions['l1'], x_positions['slm'], '200mm'),
        (x_positions['slm'], x_positions['l2'], '100mm'),
        (x_positions['l2'], x_positions['freq'], '200mm'),
        (x_positions['freq'], x_positions['l3'], '200mm'),
        (x_positions['l3'], x_positions['cam'], '200mm'),
    ]:
        ax.plot([x_start, x_end], [0.6, 0.6], [0, 0], 'b-', linewidth=2)
        ax.text((x_start+x_end)/2, 0.7, 0, label, ha='center', fontsize=8, color='blue')

    # 标题
    ax.set_title("OAM-MDNN 实验室 3D 视图\n"
                 "激光 → 扩束 → SLM(全息)→ 4f 系统(去棋盘格)→ sCMOS 采集", fontsize=13, weight='bold')
    ax.set_xlabel("光路方向 (m)")
    ax.set_ylabel("")
    ax.set_zlabel("垂直方向 (m)")
    ax.set_xlim(-0.5, 6.0)
    ax.set_ylim(-0.8, 0.8)
    ax.set_zlim(-0.6, 0.6)
    ax.view_init(elev=15, azim=-60)
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_path}")


# ==================== 原理示意图 ====================

def draw_principle(save_path="optical_setup_principle.png"):
    """画 4f 系统原理:SLM → L2 → 频谱面(pinhole 去棋盘格)→ L3 → 相机"""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    # (1) SLM 棋盘格相位
    ax1 = axes[0]
    checker = np.zeros((16, 16))
    checker[::2, ::2] = 1
    checker[1::2, 1::2] = 1
    ax1.imshow(checker, cmap='gray')
    ax1.set_title("(1) SLM 棋盘格相位\n(DPE 后)", fontsize=10)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # (2) 4f 频谱面(带 +1 级标记)
    ax2 = axes[1]
    spectrum = np.zeros((100, 100))
    # 0 级
    yy, xx = np.ogrid[:100, :100]
    spectrum += 0.8 * np.exp(-((xx-50)**2 + (yy-50)**2) / 30)
    # ±1 级
    spectrum += 0.5 * np.exp(-((xx-80)**2 + (yy-50)**2) / 25)
    spectrum += 0.5 * np.exp(-((xx-20)**2 + (yy-50)**2) / 25)
    # Pinhole
    pinhole = Circle((50, 50), 8, facecolor='red', alpha=0.4, edgecolor='red', linewidth=2)
    ax2.add_patch(pinhole)
    ax2.imshow(spectrum, cmap='hot')
    ax2.set_title("(2) 频谱面 + Pinhole\n(挡住 ±1 级)", fontsize=10)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # (3) 滤波后只剩 0 级
    ax3 = axes[2]
    ax3.imshow(spectrum * (((xx-50)**2 + (yy-50)**2) <= 8**2), cmap='hot')
    ax3.set_title("(3) Pinhole 滤波后\n(只剩 0 级)", fontsize=10)
    ax3.set_xticks([])
    ax3.set_yticks([])

    # (4) 恢复的复振幅(物光)
    ax4 = axes[3]
    # 模拟物光:振幅 + 相位
    obj_amp = np.zeros((100, 100))
    obj_amp[40:60, 40:60] = 1  # 中心亮点
    ax4.imshow(obj_amp, cmap='viridis')
    ax4.set_title("(4) 4f 输出面\n物光复振幅", fontsize=10)
    ax4.set_xticks([])
    ax4.set_yticks([])

    # 箭头连接
    for i in range(3):
        fig.add_artist(FancyArrowPatch(
            ((i+1)*4.5, 0.5), ((i+1)*4.5+0.5, 0.5),
            arrowstyle='->', mutation_scale=20, color='black', linewidth=2
        ))

    plt.suptitle("4f 系统去棋盘格原理 (棋盘格 DPE → 频谱滤波 → 复振幅恢复)",
                 fontsize=12, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_path}")


def main():
    print("=" * 60)
    print("光路图绘制")
    print("=" * 60)
    draw_2d()
    draw_3d()
    draw_principle()
    print("\n所有光路图已生成:")
    print("  - optical_setup_2d.png        俯视图(机械设计参考)")
    print("  - optical_setup_3d.png        3D 立体图(原理演示)")
    print("  - optical_setup_principle.png 4f 系统原理图(去棋盘格)")


if __name__ == "__main__":
    main()
