# -*- coding: utf-8 -*-
"""
Off-axis 数字全息 + 相位恢复
============================
从相机记录的 off-axis 全息图(含物光 + 参考光干涉)恢复复振幅。

原理:
  I(x,y) = |O + R|^2 = |O|^2 + |R|^2 + O*R* + O*R
           ---直流项---   ---干涉项 ±1 级---

  步骤:
    1. FFT(I) -> 频谱
    2. 频谱滤波: 用圆形带通取 +1 级(或 -1 级)
    3. 把 +1 级平移到频谱中心
    4. IFFT -> 复振幅 O(x,y)

参考:
  - Schnars & Juptner (2005) Digital Holography
  - Goodman (2005) Introduction to Fourier Optics
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
# scipy.ndimage.shift 可选, 代码未直接使用 (用 np.roll 实现整数平移)

# 图像读取: 优先用 imageio, 其次 tifffile, 最后 PIL
try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def read_image(path):
    """读取图像到 float32 (H, W), 自动适配 8/16 bit"""
    if HAS_IMAGEIO:
        img = imageio.imread(path)
    elif HAS_PIL:
        img = np.array(Image.open(path))
    else:
        raise ImportError("请安装 imageio 或 pillow: pip install imageio")
    return img.astype(np.float32)


# ==================== 1. 参考光生成 ====================

def generate_reference(field_shape, angle_x=0.10, angle_y=0.0, wavelength=532e-9,
                       pixel_size=8e-6, intensity=1.0):
    """
    生成 off-axis 参考光(平面波 + 倾斜相位)
    Args:
        field_shape: (H, W)
        angle_x, angle_y: 参考光角度(rad),与物光夹角
        wavelength, pixel_size: 物理参数
        intensity: 参考光强度
    Returns:
        R: (H, W) complex128
    """
    H, W = field_shape
    y = np.arange(H) * pixel_size
    x = np.arange(W) * pixel_size
    yy, xx = np.meshgrid(y, x, indexing='ij')

    # 平面波相位 = k·r = (2π/λ)(sinθx · x + sinθy · y)
    phase = (2 * np.pi / wavelength) * (np.sin(angle_x) * xx + np.sin(angle_y) * yy)
    R = np.sqrt(intensity) * np.exp(1j * phase)
    return R


# ==================== 2. Off-axis 全息图生成(模拟用) ====================

def synthesize_hologram(object_field, angle_x=0.10, angle_y=0.0,
                        ref_intensity=1.0, obj_intensity=1.0, noise=0.01):
    """
    合成 off-axis 全息图(用于测试)
    Args:
        object_field: (H, W) complex128, 物光复振幅
    Returns:
        hologram: (H, W) float32, 强度图
    """
    H, W = object_field.shape
    R = generate_reference((H, W), angle_x, angle_y)
    O = np.sqrt(obj_intensity) * object_field

    # 干涉
    hologram = np.abs(O + R) ** 2
    # 加高斯噪声模拟相机噪声
    if noise > 0:
        hologram += np.random.normal(0, noise, hologram.shape).astype(np.float32)
    return hologram.astype(np.float32), R, O


# ==================== 3. 频谱滤波(手动 / 自动) ====================

def filter_spectrum_auto(fft2_hologram, ref_angle_x=0.10, ref_angle_y=0.0,
                         wavelength=532e-9, pixel_size=8e-6,
                         filter_radius_factor=1.2):
    """
    自动定位 +1 级并滤波
    Args:
        fft2_hologram: np.fft.fft2(hologram)
        ref_angle_x, ref_angle_y: 参考光角度(与生成时一致)
        filter_radius_factor: 滤波半径 = factor × 0 级半径
    Returns:
        object_spectrum: 滤波并平移后的 +1 级频谱
    """
    H, W = fft2_hologram.shape

    # +1 级理论位置(以 0 级为原点)
    # 频谱坐标: fx = u / (W * pixel_size), x' = λ * f * fx → u = λ * f * fx / (pixel_size) / (1/(N*pixel)) = ... 简化
    # 空间频率: fx = sin(θ) / λ → 在 FFT 频谱中的索引: u0 = fx * λ * f / (pixel_size) * N
    # 对于 4f 系统, 等价于: 频谱面坐标 x' = λ * f * fx → FFT 索引 u0 = x' * N / (λ * f) (假设 f=200mm)
    # 简化: u_offset = sin(θ) * N * pixel_size / wavelength
    f1 = 200e-3  # 4f 透镜焦距(从 calc_4f_params)
    fx = np.sin(ref_angle_x) / wavelength  # cycles/m
    fy = np.sin(ref_angle_y) / wavelength

    x_offset = fx * wavelength * f1  # 频谱面位置 (m)
    y_offset = fy * wavelength * f1

    # 转 FFT 索引(假设 4f 输出面物理尺寸 = N * pixel_size)
    u_offset = int(round(x_offset / (W * pixel_size) * W))  # 索引
    v_offset = int(round(y_offset / (H * pixel_size) * H))

    # 滤波半径(0 级半宽 × factor)
    L_eff = min(H, W) * pixel_size
    radius_0 = wavelength * f1 / L_eff  # 0 级半宽 (m)
    radius_pix = max(3, int(radius_0 / (W * pixel_size) * W * filter_radius_factor))

    print(f"[Off-axis] +1 级理论位置: u={u_offset}, v={v_offset}")
    print(f"[Off-axis] 滤波半径(像素): {radius_pix}")

    # 圆型带通滤波器(中心在 +1 级位置)
    Y, X = np.ogrid[:H, :W]
    mask = ((X - u_offset) ** 2 + (Y - v_offset) ** 2) <= radius_pix ** 2

    # 提取 +1 级
    obj_spectrum = fft2_hologram * mask

    # 平移到中心(去掉载频)
    obj_spectrum_shifted = np.fft.fftshift(
        np.roll(np.roll(obj_spectrum, -v_offset, axis=0), -u_offset, axis=1)
    )
    return obj_spectrum_shifted, mask, (u_offset, v_offset, radius_pix)


def filter_spectrum_manual(fft2_hologram, peak_pos, radius_pix):
    """
    手动指定 +1 级位置(可视化拖框)
    Args:
        peak_pos: (u, v) 频谱中的 +1 级中心
        radius_pix: 滤波半径(像素)
    """
    H, W = fft2_hologram.shape
    u_offset, v_offset = peak_pos
    Y, X = np.ogrid[:H, :W]
    mask = ((X - u_offset) ** 2 + (Y - v_offset) ** 2) <= radius_pix ** 2
    obj_spectrum = fft2_hologram * mask
    obj_spectrum_shifted = np.fft.fftshift(
        np.roll(np.roll(obj_spectrum, -v_offset, axis=0), -u_offset, axis=1)
    )
    return obj_spectrum_shifted, mask


# ==================== 4. 复振幅恢复 ====================

def reconstruct_field(fft2_hologram, method="auto", **kwargs):
    """
    从频谱恢复复振幅
    Args:
        fft2_hologram: np.fft.fft2(hologram)
        method: "auto" | "manual"
    Returns:
        field: (H, W) complex128, 物光复振幅
        mask: 使用的滤波器
    """
    if method == "auto":
        obj_spectrum, mask, info = filter_spectrum_auto(fft2_hologram, **kwargs)
    elif method == "manual":
        peak_pos = kwargs.get("peak_pos")
        radius_pix = kwargs.get("radius_pix", 30)
        obj_spectrum, mask = filter_spectrum_manual(fft2_hologram, peak_pos, radius_pix)
        info = (peak_pos[0], peak_pos[1], radius_pix)
    else:
        raise ValueError(f"未知方法: {method}")

    field = np.fft.ifft2(obj_spectrum)
    return field, mask, info


# ==================== 5. 端到端测试 ====================

def test_end_to_end(save_path="off_axis_test.png"):
    """
    端到端测试:合成物光 → 合成全息 → 频谱恢复 → 验证
    """
    H, W = 1024, 1024
    rng = np.random.default_rng(42)

    # 1. 合成物光(模拟 SLM 4f 输出)
    # 中心 100×100 区域放个圆(模拟图像)
    yy, xx = np.ogrid[:H, :W]
    circle = ((yy - H//2) ** 2 + (xx - W//2) ** 2) <= 40 ** 2
    obj_amp = circle.astype(np.float32) * 0.8 + 0.1 * rng.random((H, W)).astype(np.float32)
    obj_phase = rng.uniform(-np.pi, np.pi, (H, W)).astype(np.float32)
    object_field = obj_amp * np.exp(1j * obj_phase)

    # 2. 合成 off-axis 全息图
    # 角度要确保 +1 级落在图像内: u_offset = sin(θ) * N * pixel / λ
    # 对 1024x1024, pixel 8μm, λ=532nm: u_offset = sin(θ) * 8.19e6, 取 θ=0.005 rad => u≈256
    hologram, R, O = synthesize_hologram(
        object_field, angle_x=0.005, angle_y=0.002, ref_intensity=1.0, noise=0.0
    )

    # 3. 频谱恢复
    fft2_h = np.fft.fft2(hologram)
    field_rec, mask, info = reconstruct_field(
        fft2_h, method="auto", ref_angle_x=0.005, ref_angle_y=0.002,
        wavelength=532e-9, pixel_size=8e-6, filter_radius_factor=8.0
    )

    # 4. 评估
    # 物光场 = object_field; 恢复 = field_rec (可能含全局相位差,用相关对齐)
    # 计算 PSNR
    amp_orig = np.abs(object_field)
    amp_rec = np.abs(field_rec)
    mse = np.mean((amp_orig - amp_rec) ** 2)
    psnr = 20 * np.log10(amp_orig.max() / np.sqrt(mse)) if mse > 0 else float('inf')

    print(f"\n[端到端测试] 物光 vs 恢复:")
    print(f"  振幅 MSE:  {mse:.6f}")
    print(f"  PSNR:      {psnr:.2f} dB")

    # 5. 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    axes[0, 0].imshow(hologram, cmap='gray')
    axes[0, 0].set_title("Off-axis 全息图 (强度)")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(np.log(np.abs(fft2_h) + 1), cmap='gray')
    axes[0, 1].set_title("频谱 (log scale)")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(mask, cmap='gray')
    axes[0, 2].set_title(f"滤波器 +1 级 (u,v,r)=({info[0]},{info[1]},{info[2]})")
    axes[0, 2].axis('off')

    axes[1, 0].imshow(amp_orig, cmap='gray')
    axes[1, 0].set_title("原物光振幅")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(amp_rec, cmap='gray')
    axes[1, 1].set_title(f"恢复振幅\nPSNR={psnr:.2f} dB")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(np.angle(field_rec), cmap='hsv')
    axes[1, 2].set_title("恢复相位 (角度)")
    axes[1, 2].axis('off')

    plt.suptitle("Off-axis 数字全息 + 相位恢复 (端到端测试)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n可视化已保存: {save_path}")
    return psnr


def process_real_capture(hologram_path, ref_angle_x=0.10, ref_angle_y=0.0,
                         save_path="reconstructed_field.npy"):
    """
    处理真实采集的全息图
    Args:
        hologram_path: 全息图 .tif 路径
        ref_angle_x, ref_angle_y: 实验室参考光角度(需与实验设置一致)
    """
    print(f"\n[真实数据] 加载: {hologram_path}")
    hologram = read_image(hologram_path)
    print(f"  shape: {hologram.shape}, dtype: {hologram.dtype}")

    fft2_h = np.fft.fft2(hologram)
    field_rec, mask, info = reconstruct_field(
        fft2_h, method="auto", ref_angle_x=ref_angle_x, ref_angle_y=ref_angle_y
    )

    np.save(save_path, field_rec)
    print(f"  恢复的复振幅已保存: {save_path}")
    print(f"  shape: {field_rec.shape}, dtype: {field_rec.dtype}")
    return field_rec


def main():
    # 1. 端到端测试
    print("=" * 60)
    print("Off-axis 数字全息测试")
    print("=" * 60)
    psnr = test_end_to_end()

    # 2. 处理真实数据(如果存在)
    import os
    if os.path.exists("capture.tif"):
        process_real_capture("capture.tif")
    else:
        print("\n[真实数据] 未找到 capture.tif, 跳过实物处理(可手动调用 process_real_capture)")


if __name__ == "__main__":
    main()
