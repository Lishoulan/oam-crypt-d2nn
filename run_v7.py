"""v7 完整 4 stage curriculum 训练 - 启动入口"""
import sys
import time
sys.path.insert(0, 'f:/d2nn')

# 先关闭 interactive mode
import oam_crypt_d2nn
print(f"\n[v7 启动] 4-stage curriculum 训练")
print(f"  Stage 1: 2 channels (l=-25,+25)")
print(f"  Stage 2: 5 channels (l=-25,-15,0,+15,+25)")
print(f"  Stage 3: 8 channels (l=±10,±15,±20,±25)")
print(f"  Stage 4: 10 channels (l=±5,±10,±15,±20,±25)")
print(f"  Layout: {oam_crypt_d2nn.CONFIG['layout']}")
print(f"  Curriculum: {oam_crypt_d2nn.CONFIG['curriculum']}")
print(f"  Iterative: {oam_crypt_d2nn.CONFIG['iterative_refine']}")
print(f"  OAMFreqFilter: {oam_crypt_d2nn.CONFIG['oam_freq_filter']}")
print(f"  ChannelAttention: {oam_crypt_d2nn.CONFIG['use_channel_attn']}")
print(f"  mid_ch: {oam_crypt_d2nn.CONFIG['mid_ch']}")
print(f"  num_layers: {oam_crypt_d2nn.CONFIG['num_layers']}\n")

t0 = time.time()
# 直接执行 oam_crypt_d2nn.py 的 __main__ (含 curriculum 逻辑)
exec(open('f:/d2nn/oam_crypt_d2nn.py', encoding='utf-8').read(), {"__name__": "__main__", "__file__": "f:/d2nn/oam_crypt_d2nn.py"})
elapsed = time.time() - t0
print(f"\n[v7 完成] 总耗时 {elapsed:.0f}s ({elapsed/60:.1f}min)")
