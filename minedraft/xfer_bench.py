"""GPU0 -> GPU1 transfer cost of what a disaggregated draft needs each round:
target hidden states for the accepted tokens (+ token ids), and the drafts back.

Measured with CUDA events on the destination side, rotating buffers (cold L2 on the
source is irrelevant for a peer copy of this size, but we rotate anyway).
"""
import json, sys

import torch

assert torch.cuda.device_count() >= 2
src, dst = torch.device("cuda:0"), torch.device("cuda:1")
print("p2p:", torch.cuda.can_device_access_peer(0, 1))

# hidden width per token carried to the draft (bf16):
#   EAGLE-3: 3 target layers x 4096; DFlash/DSpark on Qwen3-8B: we sweep 1..5 layers.
widths = {"1x4096": 4096, "3x4096 (EAGLE-3)": 3 * 4096, "5x4096": 5 * 4096}
rows_list = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
ROT = 8
res = []
for wname, w in widths.items():
    for rows in rows_list:
        srcs = [torch.randn(rows, w, dtype=torch.bfloat16, device=src) for _ in range(ROT)]
        dsts = [torch.empty(rows, w, dtype=torch.bfloat16, device=dst) for _ in range(ROT)]
        toks = [torch.randint(0, 150000, (rows,), device=src) for _ in range(ROT)]
        toks_d = [torch.empty(rows, dtype=torch.int64, device=dst) for _ in range(ROT)]
        for i in range(20):
            dsts[i % ROT].copy_(srcs[i % ROT], non_blocking=True)
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)
        times = []
        for it in range(50):
            j = it % ROT
            with torch.cuda.device(dst):
                e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                dsts[j].copy_(srcs[j], non_blocking=True)
                toks_d[j].copy_(toks[j], non_blocking=True)
                e1.record()
            e1.synchronize()
            times.append(e0.elapsed_time(e1))
        times.sort()
        med = times[len(times) // 2]
        mb = rows * w * 2 / 1e6
        rec = {"width": wname, "rows": rows, "MB": round(mb, 3), "median_ms": round(med, 4),
               "GBps": round(mb / 1e3 / (med / 1e3), 1)}
        res.append(rec)
        print(json.dumps(rec), flush=True)
json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "xfer.json", "w"))
