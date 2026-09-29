# 我们的两阶段索引器：实验 2
- 主注意力 Q/KV=48/4、64/8；索引器 Q/KV=4/1、8/1。
- 新方案：每 128 token 缓存一个均值 key；粗排保留 64 个块（8192 个候选 token），再精排选 2048 token。当前块强制保留，精排逐 token 因果屏蔽。
- 均值粗排为近似选择，不保证保留原始全局 top-k；未训练模型、未测任务质量。不能把性能提升直接等同于等质量加速。
- 计时包含 Q/K/V 投影、归一化、写缓存、索引投影、更新受影响块的均值、两轮评分与 top-k、分页稀疏注意力。历史块初始化不计入当前步骤。
- H200，BF16，hidden=8192，主头维度 128，索引头维度 64；最终 2048-token 总预算分块预填充；解码每请求 1 token、CUDA graph。不是整段 prompt 或整模型测量。
- 每项先预热 3 次，再采样 15 次；每次计时前在计时区外清空 L2。稠密基线逐点取 FlashInfer FA2/FA3 中较快的中位数。
- 本实验完成 384 项测量、5760 个原始样本、96 组比较、16 张柱状图；每根柱顶标出 ms。
## 胜出数量（大于 1×）
| 配置/阶段 | 原始稀疏胜出 | 两阶段胜出 | 两阶段相对稠密范围 | 两阶段快于原始稀疏 |
|---|---:|---:|---:|---:|
| 1/prefill | 2/24 | 12/24 | 0.31–3.40× | 18/24 |
| 1/decode | 9/24 | 10/24 | 0.46–6.56× | 14/24 |
| 2/prefill | 0/24 | 7/24 | 0.22–2.39× | 18/24 |
| 2/decode | 13/24 | 15/24 | 0.49–9.57× | 17/24 |
## 128k、batch 32
| 配置/阶段 | 稠密 ms | 原始稀疏 ms | 两阶段 ms | 稠密/两阶段 | 原始/两阶段 |
|---|---:|---:|---:|---:|---:|
| 1/prefill | 20.066 | 16.407 | 5.895 | 3.40× | 2.78× |
| 1/decode | 1.968 | 0.801 | 0.300 | 6.56× | 2.67× |
| 2/prefill | 26.819 | 32.049 | 11.207 | 2.39× | 2.86× |
| 2/decode | 3.958 | 1.448 | 0.414 | 9.57× | 3.50× |
## 同时加速预填充和解码
- 16k：两配置、两阶段均超过 1.10× 的 batch：无。
- 32k：两配置、两阶段均超过 1.10× 的 batch：无。
- 64k：两配置、两阶段均超过 1.10× 的 batch：32。
- 128k：两配置、两阶段均超过 1.10× 的 batch：4, 8, 16, 32。
## 候选召回检查
- 固定随机种子，batch=1，采样最后 16 个 prefill query／1 个 decode query；以下只统计门控非全零的组。指标是候选保留的全局 top-2048 token ID 比例，不是模型准确率。
- 16k：平均 55.7%，范围 53.9%–57.9%，109 个 query/group 样本。
- 128k：平均 9.3%，范围 7.6%–10.5%，109 个 query/group 样本。
- 零门控使分数全部相同，token ID 召回受并列排序影响，原始逐组数据另存。实际部署需要训练粗排策略并验证模型质量。
## 复现命令
```bash
# 在仓库根目录执行；两块空闲 H200，各跑一个配置。
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q \
  benchmarks/gqa_lightning_attention/test_attention.py \
  benchmarks/gqa_lightning_attention/test_two_stage.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/gqa_lightning_attention/benchmark_two_stage.py \
  --experiments 2 --configs 1 --contexts 16384,32768,65536,131072 \
  --batches 1,2,4,8,16,32 --phases prefill,decode \
  --variants dense_fa2,dense_fa3,sparse,two_stage --candidate-blocks 64 \
  --prefill-budget 2048 --warmup 3 --iterations 15 &
PID1=$!
CUDA_VISIBLE_DEVICES=1 .venv/bin/python benchmarks/gqa_lightning_attention/benchmark_two_stage.py \
  --experiments 2 --configs 2 --contexts 16384,32768,65536,131072 \
  --batches 1,2,4,8,16,32 --phases prefill,decode \
  --variants dense_fa2,dense_fa3,sparse,two_stage --candidate-blocks 64 \
  --prefill-budget 2048 --warmup 3 --iterations 15 &
PID2=$!
wait "$PID1"
wait "$PID2"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/gqa_lightning_attention/recall_two_stage.py --experiments 2
.venv/bin/python benchmarks/gqa_lightning_attention/report_two_stage.py --experiment 2
```
