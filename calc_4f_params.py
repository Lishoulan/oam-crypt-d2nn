# -*- coding: utf-8 -*-
"""
4f 系统参数计算器
=================
根据项目参数自动计算 4f 系统的透镜焦距、pinhole 直径、截止频率等。

输入: SLM 像素 (8 μm) + 波长 (532 nm) + 棋盘格周期
输出: 推荐 f1/f2 焦距、pinhole 直径、频谱面物理尺寸

物理原理:
  - 4f 系统的截止频率: f_c = 1 / (2 * pixel_size)  (奈奎斯特)
  - 棋盘格的空间频率: f_ck = 1 / (2 * ck_period)  (ck_period = 2*pixel_size)
  - pinhole 直径: d = λ * f / (ck_period / 2) = 2 * λ * f / ck_period
    (让低频 0 级通过, 挡住棋盘格 ±1 级)
"""

import numpy as np
import matplotlib.pyplot as plt


def calc_4f(slm_pixel=8e-6, wavelength=532e-9, ck_period_factor=2, f1=200e-3, f2=200e-3):
    """
    计算 4f 系统参数
    Args:
        slm_pixel: SLM 像素尺寸 (m)
        wavelength: 工作波长 (m)
        ck_period_factor: 棋盘格周期 = factor * slm_pixel (DPE 常用 2)
        f1, f2: 4f 透镜焦距 (m)
    Returns:
        params: dict
    """
    ck_period = ck_period_factor * slm_pixel  # 棋盘格周期
    ck_freq = 1.0 / ck_period  # 棋盘格空间频率 (cycles/m)

    # 4f 频谱面: 物面尺寸 L → 频谱面尺寸 λf / pixel
    # 频谱面坐标 x' 对应物面空间频率 fx = x' / (λ * f)
    # 物面最高频率(奈奎斯特): 1 / (2 * slm_pixel) → 频谱面位置 x'_max = λ * f / (2 * slm_pixel)
    fx_nyquist = 1.0 / (2 * slm_pixel)  # cycles/m
    x_spectrum_nyquist = wavelength * f1 * fx_nyquist  # m

    # 棋盘格 1 级频谱位置(应被 pinhole 阻挡)
    fx_ck = 1.0 / ck_period  # cycles/m
    x_spectrum_ck = wavelength * f1 * fx_ck  # m

    # 0 级频谱宽度(由 SLM 总尺寸决定)
    slm_size_x = 1920 * slm_pixel  # 15.36 mm
    slm_size_y = 1080 * slm_pixel  # 8.64 mm
    x_spectrum_0 = wavelength * f1 / slm_size_x  # 0 级半宽

    # 0 级到棋盘格 1 级的间距
    gap_0_to_ck = x_spectrum_ck - x_spectrum_0

    # 推荐 pinhole 直径: 介于 0 级和棋盘格 1 级之间
    # 经验: d = 1.5 * x_spectrum_0 (留 50% 余量), 但不大于 gap/2
    d_pinhole_recommended = min(1.5 * 2 * x_spectrum_0, 0.6 * (x_spectrum_ck - 0))

    # 输出尺寸: 4f 输出面与 SLM 物面等大(f1=f2)
    output_pixel_size = slm_pixel  # 像素尺寸不变
    output_size_x = slm_size_x
    output_size_y = slm_size_y

    params = {
        "slm_pixel": slm_pixel,
        "wavelength": wavelength,
        "ck_period": ck_period,
        "ck_period_factor": ck_period_factor,
        "f1": f1,
        "f2": f2,
        "fx_nyquist": fx_nyquist,
        "x_spectrum_nyquist": x_spectrum_nyquist,
        "fx_ck": fx_ck,
        "x_spectrum_ck": x_spectrum_ck,
        "x_spectrum_0_half": x_spectrum_0,
        "x_spectrum_0_full": 2 * x_spectrum_0,
        "gap_0_to_ck": gap_0_to_ck,
        "d_pinhole_recommended": d_pinhole_recommended,
        "output_pixel_size": output_pixel_size,
        "output_size_x": output_size_x,
        "output_size_y": output_size_y,
        "slm_size_x": slm_size_x,
        "slm_size_y": slm_size_y,
    }
    return params


def print_params(params):
    print("=" * 60)
    print("4f 系统参数计算结果 (项目标准: PLUTO SLM, 532 nm)")
    print("=" * 60)
    print(f"SLM 像素:           {params['slm_pixel']*1e6:.1f} μm")
    print(f"工作波长:           {params['wavelength']*1e9:.0f} nm")
    print(f"SLM 物理尺寸:       {params['slm_size_x']*1e3:.2f} mm × {params['slm_size_y']*1e3:.2f} mm")
    print("-" * 60)
    print(f"4f 透镜焦距:        f1 = f2 = {params['f1']*1e3:.0f} mm")
    print(f"棋盘格周期:         {params['ck_period']*1e6:.1f} μm ({params['ck_period_factor']} × 像素)")
    print(f"奈奎斯特频率:       {params['fx_nyquist']/1e3:.1f} cycles/mm")
    print(f"棋盘格空间频率:     {params['fx_ck']/1e3:.1f} cycles/mm")
    print("-" * 60)
    print(f"频谱面坐标(0 级半宽):    {params['x_spectrum_0_half']*1e3:.3f} mm")
    print(f"频谱面坐标(0 级全宽):    {params['x_spectrum_0_full']*1e3:.3f} mm")
    print(f"频谱面坐标(棋盘格 1 级): {params['x_spectrum_ck']*1e3:.3f} mm")
    print(f"0 级 → 棋盘格 1 级间距:  {params['gap_0_to_ck']*1e3:.3f} mm")
    print("=" * 60)
    print(f"★ 推荐 pinhole 直径:    {params['d_pinhole_recommended']*1e3:.3f} mm")
    print(f"  (实际: 0.1-0.2 mm, 验证后微调)")
    print("=" * 60)
    print(f"4f 输出面像素尺寸:  {params['output_pixel_size']*1e6:.1f} μm (与 SLM 相同)")
    print(f"4f 输出面物理尺寸:  {params['output_size_x']*1e3:.2f} mm × {params['output_size_y']*1e3:.2f} mm")
    print(f"相机要求:           至少 {params['output_size_x']*1e3:.0f} × {params['output_size_y']*1e3:.0f} mm 视场")
    print("=" * 60)


def plot_spectrum(params, save_path="4f_spectrum_analysis.png"):
    """画频谱面能量分布,直观看出 pinhole 位置"""
    f1 = params['f1']
    wl = params['wavelength']

    # 频谱面坐标
    x = np.linspace(-3 * params['x_spectrum_ck'], 3 * params['x_spectrum_ck'], 2000) * 1e3  # mm

    # 0 级 sinc 形 (假设均匀照明,SLM 中心 1080×1080)
    L_eff = 1080 * params['slm_pixel']  # 有效 SLM 尺寸
    I_0 = (np.sinc(x / (wl * f1 / L_eff) * 1e3)) ** 2

    # 棋盘格 ±1 级位置
    x_ck = params['x_spectrum_ck'] * 1e3  # mm
    # 1 级强度约为 0 级 60% (棋盘格 DPE 编码后)
    I_ck_plus = 0.6 * (np.sinc((x - x_ck) / (wl * f1 / L_eff) * 1e3)) ** 2
    I_ck_minus = 0.6 * (np.sinc((x + x_ck) / (wl * f1 / L_eff) * 1e3)) ** 2

    # 推荐 pinhole 位置和大小
    d_recommended = params['d_pinhole_recommended'] * 1e3  # mm

    plt.figure(figsize=(12, 6))
    plt.plot(x, I_0, 'b-', label='0 级 (低频, 含物光)', linewidth=2)
    plt.plot(x, I_ck_plus, 'r--', label='+1 级 (棋盘格高频)', linewidth=1.5)
    plt.plot(x, I_ck_minus, 'r--', label='-1 级 (棋盘格高频)', linewidth=1.5)
    plt.fill_between(x, 0, I_0,
                     where=(np.abs(x) < d_recommended/2),
                     color='green', alpha=0.3,
                     label=f'推荐 pinhole (d={d_recommended:.2f} mm)')
    plt.axvline(x_ck, color='r', linestyle=':', alpha=0.5, label=f'±1 级位置 = ±{x_ck:.2f} mm')
    plt.axvline(-x_ck, color='r', linestyle=':', alpha=0.5)
    plt.xlabel("频谱面位置 (mm)")
    plt.ylabel("归一化强度")
    plt.title(f"4f 频谱面能量分布 (f1={f1*1e3:.0f} mm, λ={wl*1e9:.0f} nm)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim(-3 * x_ck, 3 * x_ck)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n频谱分析图已保存: {save_path}")


def main():
    # 项目标准参数
    params = calc_4f(
        slm_pixel=8e-6,        # Holoeye PLUTO
        wavelength=532e-9,     # 绿光
        ck_period_factor=2,    # DPE 棋盘格周期 = 2 × 像素
        f1=200e-3,             # 第一片透镜
        f2=200e-3,             # 第二片透镜
    )
    print_params(params)
    plot_spectrum(params)

    # 验证:不同焦距的影响
    print("\n[验证] 不同焦距的 pinhole 推荐值:")
    print("-" * 60)
    print(f"{'f1 (mm)':<10} {'pinhole (mm)':<15} {'0 级宽 (mm)':<15} {'1 级位置 (mm)':<15}")
    print("-" * 60)
    for f in [100, 150, 200, 300, 500]:
        p = calc_4f(slm_pixel=8e-6, wavelength=532e-9, ck_period_factor=2,
                    f1=f*1e-3, f2=f*1e-3)
        print(f"{f:<10} {p['d_pinhole_recommended']*1e3:<15.3f} "
              f"{p['x_spectrum_0_full']*1e3:<15.3f} {p['x_spectrum_ck']*1e3:<15.3f}")
    print("-" * 60)
    print("推荐: f = 200 mm 焦距, pinhole ≈ 0.1-0.2 mm")


if __name__ == "__main__":
    main()
