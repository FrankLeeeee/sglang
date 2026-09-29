# 原生 DSA / M3 稀疏流水线调查

## 结论

- 稀疏注意力能同时加速预填充和解码，但收益取决于上下文、批量及架构；本次两种实现均未覆盖全部网格。
- DSA：预填充胜出 48/48，解码胜出 20/48。
- M3：预填充胜出 38/48，解码胜出 42/48。
- 128k 下，两配置的两阶段均超过 1.10× 的批量：DSA 为 4、8、16、32；M3 为 1、2、4、8、16、32。
- 小批量解码的索引与投影开销难以摊薄；长上下文／大批量更容易获得稳定收益。

## 方法与适用范围

- 历史实验 3 的稀疏预填充已全部胜出；本节进一步检查原生 DSA 与 M3 流水线。
- 新增 576 项测量、8640 个计时样本；两架构、两配置、两阶段、4 个上下文、6 个批量、3 种内核路径。
- 稠密基线逐点取 FA2、FA3 延迟中位数的较小值；加速比 = 最快稠密 ÷ 稀疏。
- 预填充为最后一个分块：全批新增 2048 token、eager；解码每请求 1 token、CUDA Graph。
- 均计入投影、归一化、缓存写入；稀疏还计入实际索引器、评分、top-k、索引转换及注意力。
- 隐藏维度 8192，主 Q 头数 48／64；H200、TP=1、BF16 主缓存、随机物理页；预热 3 次、采样 15 次。
- DSA：原生完整 Indexer，64Q/1KV、维度 128、FP8 索引缓存；MLA 主缓存 512+64、页长 64；选取 2048 token。
- DSA 主路径采用 rank-1536 查询投影、128+64 查询维度及 128→512 权重吸收；48 个主 Q 头填充至 64，开销计入。
- M3：主注意力 48Q/4KV 或 64Q/8KV、维度 128；索引器 4Q/1KV 或 8Q/1KV、维度 128；选 16×128 token 块，含本地块。
- M3 调用原生块评分／选择／注意力流水线；主 Q/K/V 和索引 Q/K 使用标准化投影及 RMSNorm，省略模型 RoPE 与 Gemma 增益。
- 本节是原生内核配合显式投影适配器的合成测试，并非完整模型层；不含输出投影、MLP、通信和调度。
- DSA 与 M3 的维度、页大小及语义不同；仅比较各自稠密／稀疏配对，不把跨架构绝对延迟视为同模型对比。

## 完整网格结果

| 架构 | 配置 | 阶段 | 稀疏胜出 | 加速比范围 | 样本明显分离的胜出点 |
|---|---:|---|---:|---:|---:|
| DSA | 1 | 预填充 | 24/24 | 2.44–7.31× | 24/24 |
| DSA | 1 | 解码 | 10/24 | 0.60–4.09× | 10/24 |
| DSA | 2 | 预填充 | 24/24 | 3.51–9.87× | 24/24 |
| DSA | 2 | 解码 | 10/24 | 0.61–4.18× | 10/24 |
| M3 | 1 | 预填充 | 19/24 | 0.90–4.17× | 19/24 |
| M3 | 1 | 解码 | 20/24 | 0.89–5.08× | 18/24 |
| M3 | 2 | 预填充 | 19/24 | 0.72–3.81× | 19/24 |
| M3 | 2 | 解码 | 22/24 | 0.96–9.14× | 21/24 |

- “样本明显分离”指稀疏 P90 低于两条稠密路径的最小样本；这是保守波动检查，不是统计置信区间。

## 两阶段同时更快的测试点

| 架构 | 配置 | 上下文 | 两阶段均加速的批量 | 两阶段均超过 1.10× 的批量 |
|---|---:|---:|---|---|
| DSA | 1 | 16k | 32 | 32 |
| DSA | 1 | 32k | 16、32 | 16、32 |
| DSA | 1 | 64k | 8、16、32 | 8、16、32 |
| DSA | 1 | 128k | 4、8、16、32 | 4、8、16、32 |
| DSA | 2 | 16k | 32 | 32 |
| DSA | 2 | 32k | 16、32 | 16、32 |
| DSA | 2 | 64k | 8、16、32 | 8、16、32 |
| DSA | 2 | 128k | 4、8、16、32 | 4、8、16、32 |
| M3 | 1 | 16k | 32 | 32 |
| M3 | 1 | 32k | 2、4、8、16、32 | 4、8、16、32 |
| M3 | 1 | 64k | 1、2、4、8、16、32 | 2、4、8、16、32 |
| M3 | 1 | 128k | 1、2、4、8、16、32 | 1、2、4、8、16、32 |
| M3 | 2 | 16k | 32 | 无 |
| M3 | 2 | 32k | 1、2、4、8、16、32 | 2、4、8、16、32 |
| M3 | 2 | 64k | 1、2、4、8、16、32 | 1、2、4、8、16、32 |
| M3 | 2 | 128k | 1、2、4、8、16、32 | 1、2、4、8、16、32 |

## 128k、批量 32

| 架构 | 配置 | 阶段 | 最快稠密 ms | 稀疏 ms | 加速比 | 稠密后端 |
|---|---:|---|---:|---:|---:|---|
| DSA | 1 | 预填充 | 49.405 | 8.482 | 5.82× | dense_fa3 |
| DSA | 1 | 解码 | 1.333 | 0.326 | 4.09× | dense_fa3 |
| DSA | 2 | 预填充 | 65.594 | 8.506 | 7.71× | dense_fa3 |
| DSA | 2 | 解码 | 1.361 | 0.326 | 4.18× | dense_fa3 |
| M3 | 1 | 预填充 | 19.887 | 4.769 | 4.17× | dense_fa2 |
| M3 | 1 | 解码 | 1.967 | 0.387 | 5.08× | dense_fa2 |
| M3 | 2 | 预填充 | 26.724 | 7.017 | 3.81× | dense_fa2 |
| M3 | 2 | 解码 | 3.907 | 0.427 | 9.14× | dense_fa2 |

- 不能把某些长上下文／大批量下的优势外推至全部请求；需同时检查预填充和解码。
- 随机权重只用于性能测试；块稀疏、token 稀疏及改变头数均需另做训练模型质量验证。
- 柱状图保留全部批量，柱顶标注毫秒数；纵轴从 0 开始，但图间范围不同。

## 开销定位

- DSA 在 16k／批量 1 下，单独测得索引器约 0.040–0.042 ms、稀疏注意力约 0.061–0.063 ms；解码的固定开销会抵消稀疏收益。
- DSA 在 128k／批量 32 的预填充中，单独测得索引器约 6.6 ms、稀疏注意力约 1.2–1.4 ms；索引器成为主要开销。
- M3 使用连续块，选择空间和不连续读取更少；本次长上下文下能在两个阶段获益。
- DSA 的稠密基线使用 512 维潜在表示；其预填充加速比不能直接套用于原来 128 维 GQA。
- 独立组件计时同样先预热 3 次、采样 15 次；各组件分别清刷 L2，因此不可相加为端到端延迟。

### 128k／批量 32 的预填充 CUDA 内核分解

| 架构 | 配置 | 评分 ms | 选择 ms | 稀疏注意力 ms | 其他 ms |
|---|---:|---:|---:|---:|---:|
| DSA | 1 | 4.199 | 1.023 | 1.179 | 1.486 |
| DSA | 2 | 4.207 | 1.024 | 1.191 | 1.431 |
| M3 | 1 | 2.571 | 0.109 | 1.585 | 0.451 |
| M3 | 2 | 2.993 | 0.213 | 3.131 | 0.637 |

- 此表为 eager profiler 的 CUDA 内核耗时，仅用于定位开销；预热 3 次后分析 3 次，不替代主表的无 profiler 延迟。
- 其他项包含投影、归一化、缓存读写及转换；不同架构头数／维度不同，不能把差异全部归因于块选择。
- 环境使用 NVCC 12.8；DeepGEMM 提示建议 12.9+，结果限定于本次软件环境与 BF16 主缓存路径。

## 复现命令

- 在仓库根目录执行；先安装本仓库及 PyTorch、FlashInfer、sgl-kernel、DeepGEMM、Triton、matplotlib、pytest。
- 每项测量强制至少预热 3 次；预热、JIT、自动调优与 CUDA Graph 捕获均不计入 15 个正式样本。
- 下列两条完整扫描可分别在两张空闲 H200 上运行；单卡环境将 CUDA_VISIBLE_DEVICES 都改为 0，顺序执行。

```bash
# Run from the repository root, using the benchmark environment.
# Correctness (includes causal isolation and CUDA graph replay).
CUDA_VISIBLE_DEVICES=0 BENCH_PORT=29670 .venv/bin/python -m pytest \
  benchmarks/dsa_m3_attention/test_implementations.py -q

# Full DSA sweep: 3 warmups, then 15 recorded samples per point.
CUDA_VISIBLE_DEVICES=0 BENCH_PORT=29671 .venv/bin/python \
  benchmarks/dsa_m3_attention/benchmark.py --architecture dsa \
  --configs 1,2 --contexts 16384,32768,65536,131072 --batches 1,2,4,8,16,32 \
  --phases prefill,decode --variants dense_fa2,dense_fa3,sparse \
  --prefill-budget 2048 --topk 2048 --warmup 3 --iterations 15 \
  --output benchmarks/dsa_m3_attention/results/dsa.json

# Full M3 sweep, same grid and timing contract.
CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  benchmarks/dsa_m3_attention/benchmark.py --architecture m3 \
  --configs 1,2 --contexts 16384,32768,65536,131072 --batches 1,2,4,8,16,32 \
  --phases prefill,decode --variants dense_fa2,dense_fa3,sparse \
  --prefill-budget 2048 --topk 2048 --warmup 3 --iterations 15 \
  --output benchmarks/dsa_m3_attention/results/m3.json

# Independent component timings and kernel diagnostics (also warmed up).
for arch in dsa m3; do
  CUDA_VISIBLE_DEVICES=0 BENCH_PORT=29672 .venv/bin/python \
    benchmarks/dsa_m3_attention/benchmark.py --architecture "$arch" \
    --contexts 16384,131072 --batches 1,32 --variants sparse --stages \
    --prefill-budget 2048 --topk 2048 --warmup 3 --iterations 15 \
    --output "benchmarks/dsa_m3_attention/results/${arch}_stages.json"
  CUDA_VISIBLE_DEVICES=0 BENCH_PORT=29673 .venv/bin/python \
    benchmarks/dsa_m3_attention/profile_stages.py --architecture "$arch" \
    --output "benchmarks/dsa_m3_attention/results/${arch}_profile.json"
done

# Validate the full grid; generate tables, Chinese report, and labeled bars.
.venv/bin/python benchmarks/dsa_m3_attention/report.py
```

- 主扫描输出 DSA／M3 的全部 576 项测量；诊断扫描另存，报告和 32 张带数值柱状图由原始数据生成。
- 原始样本、两个稠密后端及源码哈希见附件；中断后可用 --resume 继续同一套参数。
