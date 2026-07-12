# -*- coding: utf-8 -*-
"""
CJK 字体配置工具
================
统一设置 matplotlib 中文字体, 解决图表中文显示为方框的问题。

用法 (在脚本顶部 plt 导入之后):
    from font_config import setup_cjk
    setup_cjk()

或直接:
    import font_config  # 自动设置
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# 中文字体候选列表 (按优先级)
_CJK_FONT_CANDIDATES = [
    "SimHei",                # Windows 默认中文
    "Microsoft YaHei",       # Windows 微软雅黑
    "DengXian",              # Windows 等线
    "PingFang SC",           # macOS
    "Heiti SC",              # macOS
    "Source Han Sans CN",    # Adobe/Google 开源
    "Noto Sans CJK SC",      # Google Noto
    "WenQuanYi Micro Hei",   # Linux
    "WenQuanYi Zen Hei",     # Linux
    "Arial Unicode MS",      # 通用 unicode
    "DejaVu Sans",           # 兜底
]


def _detect_cjk_font():
    """检测系统上实际可用的 CJK 字体 (返回第一个找到的)"""
    available = {f.name for f in fm.fontManager.ttflist}
    for name in _CJK_FONT_CANDIDATES:
        if name in available:
            return name
    return None


def setup_cjk(verbose=True):
    """配置 matplotlib 支持中文显示.

    Args:
        verbose: 是否打印实际使用的字体
    Returns:
        实际生效的 CJK 字体名 (可能为 None)
    """
    cjk_font = _detect_cjk_font()
    if cjk_font is None:
        if verbose:
            print("[font_config] ⚠️  未找到 CJK 字体, 中文可能显示为方框")
            print("[font_config] 候选列表:", _CJK_FONT_CANDIDATES)
        return None

    # 重建 font.sans-serif: 把 CJK 字体放最前, 保留原 sans-serif 兜底
    current = list(plt.rcParams.get("font.sans-serif", []))
    new_list = [cjk_font] + [f for f in current if f != cjk_font]
    plt.rcParams["font.sans-serif"] = new_list
    plt.rcParams["axes.unicode_minus"] = False
    # 显式 family 防止被后续 reset
    plt.rcParams["font.family"] = "sans-serif"

    if verbose:
        print(f"[font_config] ✅ 使用 CJK 字体: {cjk_font}")
    return cjk_font


# 模块被 import 时自动设置 (兜底)
_setup_done = False


def _auto_setup():
    global _setup_done
    if not _setup_done:
        setup_cjk(verbose=False)
        _setup_done = True


_auto_setup()
