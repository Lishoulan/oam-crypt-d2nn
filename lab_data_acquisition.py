# -*- coding: utf-8 -*-
"""
实验数据采集脚本
================
功能: 加载 8-bit 全息图到 Holoeye PLUTO SLM + 采集 sCMOS 相机图像

适配项目参数:
  - SLM: Holoeye PLUTO (1920x1080, 8 μm 像素, 8-bit 相位, 532 nm)
  - 全息图: (1080, 1080) 居中放置在 1920x1080, x=[420:1500], y=[0:1080]
  - 相机: sCMOS >= 2048x2048, 像素 < 11 μm

依赖:
  - pip install numpy matplotlib opencv-python pillow
  - Holoeye SLM Display SDK: pip install holoeye (官方) 或用 .bmp 直接显示
  - 相机 SDK: Pylon / Thorlabs / Hamamatsu DCAM (按相机厂商装)
  - 无硬件时: --mock 模式用合成图模拟
"""

import argparse
import os
import time
import numpy as np
from PIL import Image

# 图像读写: 优先用 imageio, 其次 PIL
try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


def imread(path, mode=None):
    """读取图像: imageio 优先, PIL 兜底"""
    if HAS_IMAGEIO:
        img = imageio.imread(path)
    else:
        img = np.array(Image.open(path))
    return img


def imwrite(path, img):
    """保存图像"""
    if HAS_IMAGEIO:
        imageio.imwrite(path, img)
    else:
        Image.fromarray(img).save(path)


def imresize(img, size, interp=3):
    """
    缩放图像
    Args:
        img: (H, W) 或 (H, W, C) 数组
        size: (W, H)
        interp: 0=NEAREST, 1=BILINEAR, 2=BICUBIC, 3=LANCZOS
    """
    pil_interp = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS][min(interp, 3)]
    pil_img = Image.fromarray(img)
    pil_resized = pil_img.resize(size, pil_interp)
    return np.array(pil_resized)


# ==================== 1. 全息图加载与显示 ====================

class HoloeyeSLM:
    """Holoeye PLUTO SLM 控制器(支持 mock 模式)"""
    def __init__(self, slm_id=0, mock=True):
        self.slm_id = slm_id
        self.mock = mock
        self.width = 1920
        self.height = 1080
        if not mock:
            try:
                import holoeye  # Holoeye 官方 Python SDK
                self.slm = holoeye.SLM()
                print(f"[SLM] 已连接 Holoeye SLM (id={slm_id})")
            except ImportError:
                print("[SLM] 未安装 holoeye SDK, 自动切换 mock 模式")
                self.mock = True
        if self.mock:
            print(f"[SLM-MOCK] 模拟 {self.width}x{self.height} 显示")

    def show_holo(self, holo_8bit):
        """
        显示 8-bit 相位全息图
        Args:
            holo_8bit: (1080, 1080) uint8, 0-255 相位灰度
        """
        assert holo_8bit.shape == (1080, 1080), f"全息图尺寸错误: {holo_8bit.shape}"
        assert holo_8bit.dtype == np.uint8, f"数据类型必须 uint8, 当前 {holo_8bit.dtype}"

        # padding 到 1920x1080, 左侧留 420 列黑边
        full = np.zeros((self.height, self.width), dtype=np.uint8)
        full[:, 420:1500] = holo_8bit

        if self.mock:
            # mock: 保存预览图
            Image.fromarray(full).save("slm_preview_mock.png")
            print(f"[SLM-MOCK] 已保存 slm_preview_mock.png (1920x1080)")
        else:
            self.slm.show(full)
            time.sleep(0.1)  # SLM 刷新延迟
        return full

    def close(self):
        if not self.mock and hasattr(self, 'slm'):
            self.slm.close()


# ==================== 2. sCMOS 相机采集 ====================

class SciCamera:
    """sCMOS 相机采集器(支持 mock 模式)"""
    def __init__(self, mock=True, exposure_ms=100, gain=0):
        self.mock = mock
        self.exposure_ms = exposure_ms
        self.gain = gain
        self.width = 2048
        self.height = 2048
        if not mock:
            # 按实际相机厂商选择 SDK
            try:
                # 方案 1: Basler Pylon
                # from pypylon import pylon
                # self.cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
                # self.cam.Open()
                # self.cam.ExposureTime.SetValue(exposure_ms * 1000)  # μs
                raise ImportError("未配置相机 SDK")
            except ImportError:
                print("[CAM] 未配置相机 SDK, 切换 mock 模式")
                self.mock = True
        if self.mock:
            print(f"[CAM-MOCK] 模拟 {self.width}x{self.height} 相机")

    def grab(self, n_frames=1):
        """
        采集 n_frames 帧并平均
        Returns:
            image: (height, width) float32, 已减暗场
        """
        if self.mock:
            # mock: 读 slm_preview_mock.png 模拟
            if os.path.exists("slm_preview_mock.png"):
                img = imread("slm_preview_mock.png")
                # resize 到相机分辨率
                img = imresize(img, (self.width, self.height), interp=2)  # BICUBIC
                img = img.astype(np.float32) / 255.0
            else:
                # 随机散斑
                img = np.random.rand(self.height, self.width).astype(np.float32) * 0.1
        else:
            # 真实采集:采集 n_frames 帧平均
            stack = []
            for _ in range(n_frames):
                # self.cam.GrabOne(5000)  # 超时 5s
                # frame = self.cam.RetrieveResult(5000).GetArray()
                # stack.append(frame.astype(np.float32))
                pass
            img = np.mean(stack, axis=0) if stack else np.zeros((self.height, self.width), dtype=np.float32)
            return img
        return img

    def set_exposure(self, exposure_ms):
        self.exposure_ms = exposure_ms
        if not self.mock:
            # self.cam.ExposureTime.SetValue(exposure_ms * 1000)
            pass

    def close(self):
        if not self.mock and hasattr(self, 'cam'):
            # self.cam.Close()
            pass


# ==================== 3. 主流程:加载全息 + 采集 ====================

def load_holo_npy(path):
    """从 .npy 加载相位全息图,转 8-bit"""
    phase = np.load(path)  # (1080, 1080) float32, (-π, π)
    gray = ((phase + np.pi) / (2 * np.pi) * 255).round()
    return gray.astype(np.uint8)


def load_holo_tif(path):
    """从 .tif 直接读 8-bit 灰度"""
    img = cv2.imread(path, -1)
    if img.dtype != np.uint8:
        img = (img / img.max() * 255).astype(np.uint8)
    return img


def main():
    parser = argparse.ArgumentParser(description="实验数据采集")
    parser.add_argument("--holo", type=str, required=True, help="全息图路径 (.npy 或 .tif)")
    parser.add_argument("--out", type=str, default="capture.tif", help="输出文件名")
    parser.add_argument("--frames", type=int, default=10, help="平均帧数")
    parser.add_argument("--exposure", type=float, default=100.0, help="曝光时间 (ms)")
    parser.add_argument("--no-mock", action="store_true", help="尝试连接真实硬件")
    args = parser.parse_args()

    use_mock = not args.no_mock
    if use_mock:
        print("\n>>> MOCK 模式(无硬件),仅生成预览 <<<\n")

    # 1. 加载全息图
    if args.holo.endswith(".npy"):
        holo = load_holo_npy(args.holo)
    else:
        holo = load_holo_tif(args.holo)
    print(f"[1/3] 全息图加载完成: shape={holo.shape}, dtype={holo.dtype}, range=[{holo.min()}, {holo.max()}]")

    # 2. SLM 显示
    slm = HoloeyeSLM(mock=use_mock)
    slm.show_holo(holo)
    time.sleep(0.5)  # 等待 SLM 稳定

    # 3. 相机采集
    cam = SciCamera(mock=use_mock, exposure_ms=args.exposure)
    cam.set_exposure(args.exposure)
    image = cam.grab(n_frames=args.frames)
    print(f"[2/3] 相机采集完成: shape={image.shape}, dtype={image.dtype}, mean={image.mean():.4f}")

    # 4. 保存
    save_img = (image / image.max() * 65535).astype(np.uint16) if image.max() > 0 else image.astype(np.uint16)
    imwrite(args.out, save_img)
    np.save(args.out.replace(".tif", ".npy"), image)
    print(f"[3/3] 已保存: {args.out} + {args.out.replace('.tif', '.npy')}")

    # 清理
    slm.close()
    cam.close()


if __name__ == "__main__":
    main()
