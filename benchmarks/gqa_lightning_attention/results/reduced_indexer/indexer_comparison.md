# Reduced-query, shared-key indexer experiment

Configuration 1: indexer 48Q/4KV → 4Q/1KV; main attention remains 48Q/4KV.
Configuration 2: indexer 64Q/8KV → 8Q/1KV; main attention remains 64Q/8KV.

H200, BF16, TP=1, head dimension 128, indexer head dimension 64, hidden dimension 8192, exact top-2048 per attention group. All projection, norm, cache-write, scoring, top-k and attention costs are included. Each median uses 15 samples.

Prefill is a 2,048-total-token final chunk, not full-prompt latency. Decode uses one new token per request and CUDA graph replay. Previous sparse results come from `lark_rerun_20260929`; dense and reduced sparse were freshly measured together. Cross-run differences include timing noise and changed score distributions.

With one indexer query head per group, I[t,g,s] = w[t,g] * ReLU(q[t,g]·k[s]). For w>0 the gate cannot change the ranking; for w=0 all causal scores tie. The random-weight benchmark can therefore select the first 2,048 causal tokens for many groups under the smaller-index tie-break. This changes access locality and may affect timing; the experiment does not measure trained-model quality or isolate arithmetic savings alone.

## Configuration 1: prefill

Reduced sparse is faster than dense at 12/24 measured points.

| Context | Batch | Dense ms | Original sparse ms | Reduced sparse ms | Original / reduced | Dense / reduced |
|---:|---:|---:|---:|---:|---:|---:|
| 16k | 1 | 2.659 | 5.505 | 4.507 | 1.22× | 0.59× |
| 16k | 2 | 2.697 | 5.521 | 4.529 | 1.22× | 0.60× |
| 16k | 4 | 2.726 | 5.533 | 4.562 | 1.21× | 0.60× |
| 16k | 8 | 2.729 | 5.549 | 4.558 | 1.22× | 0.60× |
| 16k | 16 | 2.727 | 5.578 | 4.589 | 1.22× | 0.59× |
| 16k | 32 | 2.730 | 5.602 | 4.590 | 1.22× | 0.59× |
| 32k | 1 | 5.009 | 7.830 | 5.967 | 1.31× | 0.84× |
| 32k | 2 | 4.994 | 7.835 | 5.975 | 1.31× | 0.84× |
| 32k | 4 | 5.017 | 7.830 | 6.012 | 1.30× | 0.83× |
| 32k | 8 | 5.010 | 7.861 | 6.040 | 1.30× | 0.83× |
| 32k | 16 | 5.030 | 7.881 | 6.026 | 1.31× | 0.83× |
| 32k | 32 | 5.033 | 7.942 | 6.059 | 1.31× | 0.83× |
| 64k | 1 | 9.663 | 12.021 | 8.684 | 1.38× | 1.11× |
| 64k | 2 | 9.749 | 12.044 | 8.727 | 1.38× | 1.12× |
| 64k | 4 | 9.804 | 12.042 | 8.691 | 1.39× | 1.13× |
| 64k | 8 | 9.826 | 12.043 | 8.729 | 1.38× | 1.13× |
| 64k | 16 | 9.848 | 12.091 | 8.782 | 1.38× | 1.12× |
| 64k | 32 | 10.029 | 12.192 | 8.831 | 1.38× | 1.14× |
| 128k | 1 | 18.831 | 21.187 | 16.208 | 1.31× | 1.16× |
| 128k | 2 | 19.095 | 21.153 | 16.226 | 1.30× | 1.18× |
| 128k | 4 | 19.232 | 21.171 | 16.225 | 1.30× | 1.19× |
| 128k | 8 | 19.089 | 21.185 | 16.236 | 1.30× | 1.18× |
| 128k | 16 | 19.539 | 21.175 | 16.255 | 1.30× | 1.20× |
| 128k | 32 | 19.749 | 21.422 | 16.299 | 1.31× | 1.21× |

![Configuration 1, prefill, 16k](indexer_charts/config1_prefill_16k.png)

![Configuration 1, prefill, 32k](indexer_charts/config1_prefill_32k.png)

![Configuration 1, prefill, 64k](indexer_charts/config1_prefill_64k.png)

![Configuration 1, prefill, 128k](indexer_charts/config1_prefill_128k.png)

## Configuration 1: decode

Reduced sparse is faster than dense at 9/24 measured points.

| Context | Batch | Dense ms | Original sparse ms | Reduced sparse ms | Original / reduced | Dense / reduced |
|---:|---:|---:|---:|---:|---:|---:|
| 16k | 1 | 0.079 | 0.136 | 0.121 | 1.13× | 0.66× |
| 16k | 2 | 0.085 | 0.154 | 0.155 | 0.99× | 0.55× |
| 16k | 4 | 0.104 | 0.171 | 0.171 | 1.00× | 0.61× |
| 16k | 8 | 0.134 | 0.202 | 0.198 | 1.02× | 0.68× |
| 16k | 16 | 0.195 | 0.261 | 0.241 | 1.08× | 0.81× |
| 16k | 32 | 0.313 | 0.300 | 0.272 | 1.10× | 1.15× |
| 32k | 1 | 0.085 | 0.157 | 0.141 | 1.12× | 0.61× |
| 32k | 2 | 0.102 | 0.172 | 0.183 | 0.94× | 0.56× |
| 32k | 4 | 0.134 | 0.190 | 0.200 | 0.95× | 0.67× |
| 32k | 8 | 0.193 | 0.229 | 0.233 | 0.98× | 0.83× |
| 32k | 16 | 0.313 | 0.305 | 0.303 | 1.01× | 1.03× |
| 32k | 32 | 0.552 | 0.377 | 0.359 | 1.05× | 1.53× |
| 64k | 1 | 0.104 | 0.177 | 0.158 | 1.12× | 0.66× |
| 64k | 2 | 0.136 | 0.193 | 0.232 | 0.83× | 0.59× |
| 64k | 4 | 0.193 | 0.223 | 0.253 | 0.88× | 0.76× |
| 64k | 8 | 0.311 | 0.278 | 0.300 | 0.93× | 1.04× |
| 64k | 16 | 0.548 | 0.387 | 0.396 | 0.98× | 1.38× |
| 64k | 32 | 1.025 | 0.528 | 0.503 | 1.05× | 2.04× |
| 128k | 1 | 0.136 | 0.209 | 0.185 | 1.13× | 0.73× |
| 128k | 2 | 0.191 | 0.234 | 0.318 | 0.73× | 0.60× |
| 128k | 4 | 0.311 | 0.277 | 0.356 | 0.78× | 0.87× |
| 128k | 8 | 0.547 | 0.367 | 0.431 | 0.85× | 1.27× |
| 128k | 16 | 1.022 | 0.545 | 0.567 | 0.96× | 1.80× |
| 128k | 32 | 1.962 | 0.827 | 0.789 | 1.05× | 2.49× |

![Configuration 1, decode, 16k](indexer_charts/config1_decode_16k.png)

![Configuration 1, decode, 32k](indexer_charts/config1_decode_32k.png)

![Configuration 1, decode, 64k](indexer_charts/config1_decode_64k.png)

![Configuration 1, decode, 128k](indexer_charts/config1_decode_128k.png)

## Configuration 2: prefill

Reduced sparse is faster than dense at 0/24 measured points.

| Context | Batch | Dense ms | Original sparse ms | Reduced sparse ms | Original / reduced | Dense / reduced |
|---:|---:|---:|---:|---:|---:|---:|
| 16k | 1 | 3.548 | 9.899 | 8.644 | 1.15× | 0.41× |
| 16k | 2 | 3.606 | 9.972 | 8.758 | 1.14× | 0.41× |
| 16k | 4 | 3.633 | 9.975 | 8.752 | 1.14× | 0.42× |
| 16k | 8 | 3.650 | 10.034 | 8.784 | 1.14× | 0.42× |
| 16k | 16 | 3.651 | 10.113 | 8.813 | 1.15× | 0.41× |
| 16k | 32 | 3.671 | 10.178 | 8.875 | 1.15× | 0.41× |
| 32k | 1 | 6.589 | 13.978 | 11.614 | 1.20× | 0.57× |
| 32k | 2 | 6.668 | 13.984 | 11.704 | 1.19× | 0.57× |
| 32k | 4 | 6.682 | 13.993 | 11.707 | 1.20× | 0.57× |
| 32k | 8 | 6.821 | 14.010 | 11.719 | 1.20× | 0.58× |
| 32k | 16 | 6.960 | 14.102 | 11.745 | 1.20× | 0.59× |
| 32k | 32 | 7.140 | 14.256 | 11.813 | 1.21× | 0.60× |
| 64k | 1 | 13.099 | 21.516 | 17.048 | 1.26× | 0.77× |
| 64k | 2 | 13.145 | 21.419 | 17.158 | 1.25× | 0.77× |
| 64k | 4 | 13.076 | 21.455 | 17.146 | 1.25× | 0.76× |
| 64k | 8 | 13.304 | 21.444 | 17.065 | 1.26× | 0.78× |
| 64k | 16 | 13.445 | 21.435 | 17.170 | 1.25× | 0.78× |
| 64k | 32 | 13.609 | 21.641 | 17.267 | 1.25× | 0.79× |
| 128k | 1 | 25.479 | 38.642 | 32.106 | 1.20× | 0.79× |
| 128k | 2 | 25.787 | 38.600 | 32.035 | 1.20× | 0.80× |
| 128k | 4 | 25.734 | 38.604 | 32.073 | 1.20× | 0.80× |
| 128k | 8 | 25.970 | 38.581 | 32.090 | 1.20× | 0.81× |
| 128k | 16 | 26.291 | 38.579 | 32.073 | 1.20× | 0.82× |
| 128k | 32 | 26.623 | 38.575 | 32.029 | 1.20× | 0.83× |

![Configuration 2, prefill, 16k](indexer_charts/config2_prefill_16k.png)

![Configuration 2, prefill, 32k](indexer_charts/config2_prefill_32k.png)

![Configuration 2, prefill, 64k](indexer_charts/config2_prefill_64k.png)

![Configuration 2, prefill, 128k](indexer_charts/config2_prefill_128k.png)

## Configuration 2: decode

Reduced sparse is faster than dense at 13/24 measured points.

| Context | Batch | Dense ms | Original sparse ms | Reduced sparse ms | Original / reduced | Dense / reduced |
|---:|---:|---:|---:|---:|---:|---:|
| 16k | 1 | 0.096 | 0.157 | 0.152 | 1.03× | 0.63× |
| 16k | 2 | 0.111 | 0.181 | 0.163 | 1.11× | 0.68× |
| 16k | 4 | 0.140 | 0.204 | 0.193 | 1.06× | 0.73× |
| 16k | 8 | 0.205 | 0.248 | 0.222 | 1.12× | 0.92× |
| 16k | 16 | 0.322 | 0.307 | 0.273 | 1.13× | 1.18× |
| 16k | 32 | 0.568 | 0.446 | 0.406 | 1.10× | 1.40× |
| 32k | 1 | 0.111 | 0.181 | 0.180 | 1.01× | 0.62× |
| 32k | 2 | 0.140 | 0.198 | 0.193 | 1.03× | 0.73× |
| 32k | 4 | 0.200 | 0.229 | 0.219 | 1.05× | 0.91× |
| 32k | 8 | 0.320 | 0.291 | 0.282 | 1.03× | 1.14× |
| 32k | 16 | 0.564 | 0.384 | 0.361 | 1.06× | 1.57× |
| 32k | 32 | 1.051 | 0.596 | 0.555 | 1.07× | 1.89× |
| 64k | 1 | 0.140 | 0.204 | 0.236 | 0.86× | 0.59× |
| 64k | 2 | 0.204 | 0.228 | 0.255 | 0.89× | 0.80× |
| 64k | 4 | 0.322 | 0.278 | 0.295 | 0.94× | 1.09× |
| 64k | 8 | 0.563 | 0.372 | 0.373 | 1.00× | 1.51× |
| 64k | 16 | 1.043 | 0.534 | 0.503 | 1.06× | 2.07× |
| 64k | 32 | 2.011 | 0.893 | 0.850 | 1.05× | 2.36× |
| 128k | 1 | 0.203 | 0.241 | 0.326 | 0.74× | 0.62× |
| 128k | 2 | 0.322 | 0.284 | 0.357 | 0.80× | 0.90× |
| 128k | 4 | 0.561 | 0.365 | 0.423 | 0.86× | 1.33× |
| 128k | 8 | 1.044 | 0.528 | 0.555 | 0.95× | 1.88× |
| 128k | 16 | 1.997 | 0.833 | 0.801 | 1.04× | 2.49× |
| 128k | 32 | 3.954 | 1.490 | 1.442 | 1.03× | 2.74× |

![Configuration 2, decode, 16k](indexer_charts/config2_decode_16k.png)

![Configuration 2, decode, 32k](indexer_charts/config2_decode_32k.png)

![Configuration 2, decode, 64k](indexer_charts/config2_decode_64k.png)

![Configuration 2, decode, 128k](indexer_charts/config2_decode_128k.png)
