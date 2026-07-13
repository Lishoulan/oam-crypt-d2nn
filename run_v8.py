# -*- coding: utf-8 -*-
"""
v8 训练脚本: 4 stage curriculum (2->5->8->10 通道) + PolarConv + OAM-FDD loss
启动: py run_v8.py
"""
import os
import sys
import time
import numpy as np
import torch

# 添加当前目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oam_crypt_d2nn import (
    CONFIG, OAM_Crypt_D2NN,
    train_one_stage, generate_rpp,
)
from font_config import setup_cjk
setup_cjk()  # 确保 matplotlib 中文显示正常

print("="*80)
print("[v8] PolarHNN (Polar Holographic Neural Network) 训练")
print("="*80)
print(f"[v8] 创新 1: PolarConv (极坐标卷积): {CONFIG.get('polar_conv')}")
print(f"[v8] 创新 2: OAM-FDD Loss: {CONFIG.get('oam_fdd_loss')}")
print(f"[v8] 创新 3: Multi-scale OAM: {CONFIG.get('multi_scale_oam')}")
print(f"[v8] layout: {CONFIG['layout']}")
print(f"[v8] polar_n_r: {CONFIG.get('polar_n_r')}, polar_n_theta: {CONFIG.get('polar_n_theta')}")
print(f"[v8] curriculum: {CONFIG.get('curriculum')}")

if __name__ == "__main__":
    device = torch.device(CONFIG['device'])
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
        print(f"\n[v8 oam_overlap] 自适应覆盖: epochs={CONFIG['epochs']} mid_ch={CONFIG['mid_ch']} "
              f"num_layers={CONFIG['num_layers']} channel_attn={CONFIG['use_channel_attn']}", flush=True)
        print(f"[v8] polar_conv: {CONFIG.get('polar_conv')}", flush=True)

    # ---------- 1. 全局 RPP (跨 stage 共享) ----------
    os.makedirs("./data", exist_ok=True)
    rpp_system = generate_rpp(CONFIG["size"], device)

    # ---------- 2. Curriculum Learning 4 stage ----------
    if CONFIG.get("curriculum", True):
        stages = CONFIG["curriculum_stages"]
        n_stages = len(stages)
        threshold = CONFIG["curriculum_psnr_threshold"]
        print(f"\n[v8 CURRICULUM] {n_stages} stages, PSNR_C threshold={threshold} dB", flush=True)

        stage_results = []
        t_start_total = time.time()
        for s_idx, stage_cfg in enumerate(stages):
            t_start_stage = time.time()
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
            stage_time = time.time() - t_start_stage
            stage_results.append((s_idx, stage_cfg["n_channels"], best_psnr, best_sr, stage_time))
            torch.save({
                'stage': s_idx + 1,
                'n_channels': stage_cfg["n_channels"],
                'l_auth': stage_cfg["l_auth"],
                'z_list': stage_cfg["z_list"],
                'model_state_dict': best_state,
                'psnr_center': best_psnr,
                'sec_ratio': best_sr,
            }, f"oam_crypt_v8_stage{s_idx+1}_best.pth")
            print(f"  [v8 STAGE {s_idx+1} DONE] n_ch={stage_cfg['n_channels']} "
                  f"PSNR_C={best_psnr:.2f} dB SR_OAM={best_sr:.4f} "
                  f"t={stage_time:.0f}s saved to oam_crypt_v8_stage{s_idx+1}_best.pth", flush=True)
            if s_idx < n_stages - 1 and best_psnr < threshold:
                print(f"  [v8 WARNING] Stage {s_idx+1} PSNR_C={best_psnr:.2f} < threshold {threshold}, "
                      f"continuing to next stage (curriculum strategy)", flush=True)

        # 加载最后 stage 最佳模型作为最终 v8 模型
        final_ckpt = torch.load(f"oam_crypt_v8_stage{n_stages}_best.pth", map_location=device)
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
            iterative_refine=CONFIG.get("iterative_refine", False),
            n_passes=CONFIG.get("n_passes", 3),
            oam_freq_filter=CONFIG.get("oam_freq_filter", True),
            use_polar_conv=CONFIG.get("polar_conv", True),
            polar_n_r=CONFIG.get("polar_n_r", 32),
            polar_n_theta=CONFIG.get("polar_n_theta", 96),
        ).to(device)
        final_model.load_state_dict(final_ckpt['model_state_dict'])
        final_model.eval()

        total_time = time.time() - t_start_total
        torch.save({
            'version': 'v8',
            'model_state_dict': final_model.state_dict(),
            'stage_results': stage_results,
            'psnr_center': final_ckpt['psnr_center'],
            'sec_ratio': final_ckpt['sec_ratio'],
            'total_time': total_time,
        }, "oam_crypt_v8_final.pth")
        print(f"\n[v8 FINAL] PSNR_C={final_ckpt['psnr_center']:.2f} dB "
              f"SR_OAM={final_ckpt['sec_ratio']:.4f} "
              f"total_time={total_time/60:.1f}min "
              f"saved to oam_crypt_v8_final.pth", flush=True)

        # Stage 总结
        print(f"\n[v8 SUMMARY] Curriculum 4 stage 训练总结:")
        for s_idx, n_ch, psnr, sr, t_sec in stage_results:
            print(f"  Stage {s_idx+1}: n_ch={n_ch:2d} PSNR_C={psnr:.2f} dB SR_OAM={sr:.4f} t={t_sec:.0f}s")
    else:
        print("[v8 WARNING] curriculum=False, 退化到单 stage 训练", flush=True)

    print("\n[v8] 训练流程结束")
