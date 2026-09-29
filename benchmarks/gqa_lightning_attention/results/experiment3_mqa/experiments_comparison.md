# Experiment 3: main attention with one KV head

Only the main attention KV head count changes from experiment 2. Indexer query heads remain 4/8 and indexer KV heads remain 1. Main Q heads remain 48/64.

| Experiment | Main Q/KV (config 1; config 2) | Indexer Q/KV (config 1; config 2) |
|---|---|---|
| 1 | 48/4; 64/8 | 48/4; 64/8 |
| 2 | 48/4; 64/8 | 4/1; 8/1 |
| 3 | 48/1; 64/1 | 4/1; 8/1 |

H200, BF16, TP=1, hidden dimension 8192, main head dimension 128, indexer head dimension 64, top-2048, page size 128. Prefill is a final chunk of 2,048 total new tokens across the batch. Decode is one new token/request with CUDA graph replay. Latencies include projections, normalization, cache writes, scores, selection and attention. Each median uses 15 CUDA-event samples.

In experiment 3 there is one attention KV group: the 4/8 weighted indexer heads are summed into one score per token, and one top-2048 set serves all 48/64 main Q heads. Experiment 2 instead has 4/8 separate scores and selections. This reduces both cache size and selection count; it also changes model behavior.

Dense MQA uses the same FlashInfer paged backend, with a 1 GiB workspace required by its tensor-core decode planner (128 MiB for the previous GQA experiments). Workspace allocation and planning are outside timing.

Experiment 1 and 2 use their existing measured runs. Experiment 3 is a fresh complete 192-measurement run, not an extrapolation. These synthetic random-weight timings do not establish model quality. Different shapes also change random weight/cache realizations, and cross-run comparisons include timing noise.

[Experiment 3 dense vs sparse tables](comparison.md) · [All three experiments CSV](experiments_comparison.csv)

## Configuration 1: prefill

Experiment 3 sparse is faster than its dense baseline at 24/24 measured points.

| Context | Batch | E1 dense | E1 sparse | E2 dense | E2 sparse | E3 dense | E3 sparse | E3 dense/sparse |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16k | 1 | 2.659 | 5.505 | 2.659 | 4.507 | 2.585 | 2.052 | 1.26× |
| 16k | 2 | 2.691 | 5.521 | 2.697 | 4.529 | 2.649 | 2.057 | 1.29× |
| 16k | 4 | 2.712 | 5.533 | 2.726 | 4.562 | 2.678 | 2.061 | 1.30× |
| 16k | 8 | 2.727 | 5.549 | 2.729 | 4.558 | 2.699 | 2.080 | 1.30× |
| 16k | 16 | 2.737 | 5.578 | 2.727 | 4.589 | 2.700 | 2.074 | 1.30× |
| 16k | 32 | 2.740 | 5.602 | 2.730 | 4.590 | 2.699 | 2.098 | 1.29× |
| 32k | 1 | 4.956 | 7.830 | 5.009 | 5.967 | 4.871 | 2.439 | 2.00× |
| 32k | 2 | 5.013 | 7.835 | 4.994 | 5.975 | 4.944 | 2.458 | 2.01× |
| 32k | 4 | 5.008 | 7.830 | 5.017 | 6.012 | 4.974 | 2.469 | 2.01× |
| 32k | 8 | 5.011 | 7.861 | 5.010 | 6.040 | 4.977 | 2.467 | 2.02× |
| 32k | 16 | 5.051 | 7.881 | 5.030 | 6.026 | 5.006 | 2.486 | 2.01× |
| 32k | 32 | 5.120 | 7.942 | 5.033 | 6.059 | 4.985 | 2.495 | 2.00× |
| 64k | 1 | 9.563 | 12.021 | 9.663 | 8.684 | 9.622 | 3.304 | 2.91× |
| 64k | 2 | 9.729 | 12.044 | 9.749 | 8.727 | 9.680 | 3.335 | 2.90× |
| 64k | 4 | 9.823 | 12.042 | 9.804 | 8.691 | 9.580 | 3.344 | 2.86× |
| 64k | 8 | 9.757 | 12.043 | 9.826 | 8.729 | 9.705 | 3.355 | 2.89× |
| 64k | 16 | 9.905 | 12.091 | 9.848 | 8.782 | 9.717 | 3.371 | 2.88× |
| 64k | 32 | 10.082 | 12.192 | 10.029 | 8.831 | 9.793 | 3.390 | 2.89× |
| 128k | 1 | 18.826 | 21.187 | 18.831 | 16.208 | 18.752 | 5.563 | 3.37× |
| 128k | 2 | 19.177 | 21.153 | 19.095 | 16.226 | 18.983 | 5.601 | 3.39× |
| 128k | 4 | 19.371 | 21.171 | 19.232 | 16.225 | 19.037 | 5.618 | 3.39× |
| 128k | 8 | 19.439 | 21.185 | 19.089 | 16.236 | 19.172 | 5.625 | 3.41× |
| 128k | 16 | 19.549 | 21.175 | 19.539 | 16.255 | 19.360 | 5.621 | 3.44× |
| 128k | 32 | 19.888 | 21.422 | 19.749 | 16.299 | 19.060 | 5.649 | 3.37× |

All latencies are milliseconds; ratios above 1 mean sparse is faster.

![Experiment 3 config 1 prefill 16k](charts/config1_prefill_16k.png)

![Experiment 3 config 1 prefill 32k](charts/config1_prefill_32k.png)

![Experiment 3 config 1 prefill 64k](charts/config1_prefill_64k.png)

![Experiment 3 config 1 prefill 128k](charts/config1_prefill_128k.png)

## Configuration 1: decode

Experiment 3 sparse is faster than its dense baseline at 1/24 measured points.

| Context | Batch | E1 dense | E1 sparse | E2 dense | E2 sparse | E3 dense | E3 sparse | E3 dense/sparse |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16k | 1 | 0.077 | 0.136 | 0.079 | 0.121 | 0.070 | 0.117 | 0.60× |
| 16k | 2 | 0.085 | 0.154 | 0.085 | 0.155 | 0.073 | 0.146 | 0.50× |
| 16k | 4 | 0.103 | 0.171 | 0.104 | 0.171 | 0.080 | 0.153 | 0.52× |
| 16k | 8 | 0.135 | 0.202 | 0.134 | 0.198 | 0.088 | 0.160 | 0.55× |
| 16k | 16 | 0.195 | 0.261 | 0.195 | 0.241 | 0.105 | 0.182 | 0.58× |
| 16k | 32 | 0.317 | 0.300 | 0.313 | 0.272 | 0.141 | 0.224 | 0.63× |
| 32k | 1 | 0.090 | 0.157 | 0.085 | 0.141 | 0.081 | 0.136 | 0.59× |
| 32k | 2 | 0.105 | 0.172 | 0.102 | 0.183 | 0.082 | 0.173 | 0.47× |
| 32k | 4 | 0.134 | 0.190 | 0.134 | 0.200 | 0.090 | 0.179 | 0.50× |
| 32k | 8 | 0.194 | 0.229 | 0.193 | 0.233 | 0.103 | 0.189 | 0.55× |
| 32k | 16 | 0.312 | 0.305 | 0.313 | 0.303 | 0.138 | 0.215 | 0.64× |
| 32k | 32 | 0.552 | 0.377 | 0.552 | 0.359 | 0.203 | 0.265 | 0.77× |
| 64k | 1 | 0.105 | 0.177 | 0.104 | 0.158 | 0.085 | 0.149 | 0.57× |
| 64k | 2 | 0.134 | 0.193 | 0.136 | 0.232 | 0.091 | 0.216 | 0.42× |
| 64k | 4 | 0.194 | 0.223 | 0.193 | 0.253 | 0.104 | 0.225 | 0.46× |
| 64k | 8 | 0.311 | 0.278 | 0.311 | 0.300 | 0.137 | 0.237 | 0.57× |
| 64k | 16 | 0.548 | 0.387 | 0.548 | 0.396 | 0.202 | 0.272 | 0.74× |
| 64k | 32 | 1.024 | 0.528 | 1.025 | 0.503 | 0.327 | 0.340 | 0.96× |
| 128k | 1 | 0.135 | 0.209 | 0.136 | 0.185 | 0.092 | 0.168 | 0.55× |
| 128k | 2 | 0.195 | 0.234 | 0.191 | 0.318 | 0.107 | 0.296 | 0.36× |
| 128k | 4 | 0.310 | 0.277 | 0.311 | 0.356 | 0.135 | 0.308 | 0.44× |
| 128k | 8 | 0.548 | 0.367 | 0.547 | 0.431 | 0.201 | 0.332 | 0.60× |
| 128k | 16 | 1.020 | 0.545 | 1.022 | 0.567 | 0.324 | 0.383 | 0.85× |
| 128k | 32 | 1.966 | 0.827 | 1.962 | 0.789 | 0.571 | 0.483 | 1.18× |

All latencies are milliseconds; ratios above 1 mean sparse is faster.

![Experiment 3 config 1 decode 16k](charts/config1_decode_16k.png)

![Experiment 3 config 1 decode 32k](charts/config1_decode_32k.png)

![Experiment 3 config 1 decode 64k](charts/config1_decode_64k.png)

![Experiment 3 config 1 decode 128k](charts/config1_decode_128k.png)

## Configuration 2: prefill

Experiment 3 sparse is faster than its dense baseline at 24/24 measured points.

| Context | Batch | E1 dense | E1 sparse | E2 dense | E2 sparse | E3 dense | E3 sparse | E3 dense/sparse |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16k | 1 | 3.556 | 9.899 | 3.548 | 8.644 | 3.414 | 2.209 | 1.55× |
| 16k | 2 | 3.605 | 9.972 | 3.606 | 8.758 | 3.504 | 2.234 | 1.57× |
| 16k | 4 | 3.626 | 9.975 | 3.633 | 8.752 | 3.560 | 2.250 | 1.58× |
| 16k | 8 | 3.650 | 10.034 | 3.650 | 8.784 | 3.558 | 2.274 | 1.56× |
| 16k | 16 | 3.651 | 10.113 | 3.651 | 8.813 | 3.592 | 2.294 | 1.57× |
| 16k | 32 | 3.719 | 10.178 | 3.671 | 8.875 | 3.601 | 2.327 | 1.55× |
| 32k | 1 | 6.639 | 13.978 | 6.589 | 11.614 | 6.498 | 2.699 | 2.41× |
| 32k | 2 | 6.737 | 13.984 | 6.668 | 11.704 | 6.640 | 2.708 | 2.45× |
| 32k | 4 | 6.711 | 13.993 | 6.682 | 11.707 | 6.693 | 2.726 | 2.46× |
| 32k | 8 | 6.844 | 14.010 | 6.821 | 11.719 | 6.616 | 2.746 | 2.41× |
| 32k | 16 | 6.765 | 14.102 | 6.960 | 11.745 | 6.764 | 2.763 | 2.45× |
| 32k | 32 | 6.967 | 14.256 | 7.140 | 11.813 | 6.717 | 2.784 | 2.41× |
| 64k | 1 | 13.087 | 21.516 | 13.099 | 17.048 | 12.745 | 3.640 | 3.50× |
| 64k | 2 | 13.151 | 21.419 | 13.145 | 17.158 | 12.834 | 3.644 | 3.52× |
| 64k | 4 | 13.110 | 21.455 | 13.076 | 17.146 | 13.100 | 3.654 | 3.59× |
| 64k | 8 | 13.308 | 21.444 | 13.304 | 17.065 | 12.979 | 3.680 | 3.53× |
| 64k | 16 | 13.448 | 21.435 | 13.445 | 17.170 | 13.019 | 3.726 | 3.49× |
| 64k | 32 | 13.668 | 21.641 | 13.609 | 17.267 | 13.238 | 3.748 | 3.53× |
| 128k | 1 | 25.486 | 38.642 | 25.479 | 32.106 | 25.188 | 5.880 | 4.28× |
| 128k | 2 | 25.824 | 38.600 | 25.787 | 32.035 | 25.431 | 5.901 | 4.31× |
| 128k | 4 | 25.955 | 38.604 | 25.734 | 32.073 | 25.450 | 5.891 | 4.32× |
| 128k | 8 | 26.013 | 38.581 | 25.970 | 32.090 | 25.521 | 5.963 | 4.28× |
| 128k | 16 | 26.292 | 38.579 | 26.291 | 32.073 | 25.526 | 6.005 | 4.25× |
| 128k | 32 | 26.809 | 38.575 | 26.623 | 32.029 | 25.695 | 6.032 | 4.26× |

All latencies are milliseconds; ratios above 1 mean sparse is faster.

![Experiment 3 config 2 prefill 16k](charts/config2_prefill_16k.png)

![Experiment 3 config 2 prefill 32k](charts/config2_prefill_32k.png)

![Experiment 3 config 2 prefill 64k](charts/config2_prefill_64k.png)

![Experiment 3 config 2 prefill 128k](charts/config2_prefill_128k.png)

## Configuration 2: decode

Experiment 3 sparse is faster than its dense baseline at 3/24 measured points.

| Context | Batch | E1 dense | E1 sparse | E2 dense | E2 sparse | E3 dense | E3 sparse | E3 dense/sparse |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16k | 1 | 0.096 | 0.157 | 0.096 | 0.152 | 0.078 | 0.128 | 0.61× |
| 16k | 2 | 0.111 | 0.181 | 0.111 | 0.163 | 0.080 | 0.131 | 0.61× |
| 16k | 4 | 0.144 | 0.204 | 0.140 | 0.193 | 0.087 | 0.146 | 0.60× |
| 16k | 8 | 0.205 | 0.248 | 0.205 | 0.222 | 0.096 | 0.157 | 0.61× |
| 16k | 16 | 0.324 | 0.307 | 0.322 | 0.273 | 0.112 | 0.177 | 0.64× |
| 16k | 32 | 0.572 | 0.446 | 0.568 | 0.406 | 0.149 | 0.224 | 0.66× |
| 32k | 1 | 0.114 | 0.181 | 0.111 | 0.180 | 0.086 | 0.146 | 0.59× |
| 32k | 2 | 0.144 | 0.198 | 0.140 | 0.193 | 0.088 | 0.150 | 0.58× |
| 32k | 4 | 0.203 | 0.229 | 0.200 | 0.219 | 0.095 | 0.155 | 0.61× |
| 32k | 8 | 0.323 | 0.291 | 0.320 | 0.282 | 0.112 | 0.169 | 0.66× |
| 32k | 16 | 0.564 | 0.384 | 0.564 | 0.361 | 0.146 | 0.195 | 0.75× |
| 32k | 32 | 1.052 | 0.596 | 1.051 | 0.555 | 0.215 | 0.250 | 0.86× |
| 64k | 1 | 0.142 | 0.204 | 0.140 | 0.236 | 0.090 | 0.162 | 0.55× |
| 64k | 2 | 0.204 | 0.228 | 0.204 | 0.255 | 0.096 | 0.166 | 0.58× |
| 64k | 4 | 0.323 | 0.278 | 0.322 | 0.295 | 0.111 | 0.172 | 0.64× |
| 64k | 8 | 0.562 | 0.372 | 0.563 | 0.373 | 0.143 | 0.191 | 0.75× |
| 64k | 16 | 1.041 | 0.534 | 1.043 | 0.503 | 0.211 | 0.225 | 0.94× |
| 64k | 32 | 2.013 | 0.893 | 2.011 | 0.850 | 0.337 | 0.298 | 1.13× |
| 128k | 1 | 0.202 | 0.241 | 0.203 | 0.326 | 0.098 | 0.179 | 0.54× |
| 128k | 2 | 0.323 | 0.284 | 0.322 | 0.357 | 0.111 | 0.184 | 0.60× |
| 128k | 4 | 0.562 | 0.365 | 0.561 | 0.423 | 0.142 | 0.203 | 0.70× |
| 128k | 8 | 1.041 | 0.528 | 1.044 | 0.555 | 0.209 | 0.231 | 0.91× |
| 128k | 16 | 1.994 | 0.833 | 1.997 | 0.801 | 0.333 | 0.283 | 1.18× |
| 128k | 32 | 3.960 | 1.490 | 3.954 | 1.442 | 0.587 | 0.377 | 1.56× |

All latencies are milliseconds; ratios above 1 mean sparse is faster.

![Experiment 3 config 2 decode 16k](charts/config2_decode_16k.png)

![Experiment 3 config 2 decode 32k](charts/config2_decode_32k.png)

![Experiment 3 config 2 decode 64k](charts/config2_decode_64k.png)

![Experiment 3 config 2 decode 128k](charts/config2_decode_128k.png)
