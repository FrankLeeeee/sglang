# Final sparse-attention study

1440 sparse measurements.

## Wins over the best dense baseline

| Method | Indexer | Dim | Prefill wins | Decode wins |
|---|---|---:|---:|---:|
| token | identical | 64 | 0/58 | 39/58 |
| token | identical | 128 | 0/58 | 29/58 |
| token | reduced | 64 | 0/60 | 35/60 |
| token | reduced | 128 | 0/60 | 32/60 |
| block | identical | 64 | 9/58 | 42/58 |
| block | identical | 128 | 0/58 | 36/58 |
| block | reduced | 64 | 48/60 | 46/60 |
| block | reduced | 128 | 45/60 | 42/60 |
| two_stage | identical | 64 | 31/58 | 40/58 |
| two_stage | identical | 128 | 25/58 | 38/58 |
| two_stage | reduced | 64 | 37/60 | 42/60 |
| two_stage | reduced | 128 | 33/60 | 42/60 |

## Best sparse speedup per point (T token, B block, 2 two-stage)

| Main | Phase | Batch | 16k | 64k | 128k | 512k | 1024k |
|---|---|---:|---:|---:|---:|---:|---:|
| 48q4kv | prefill | 1 | 0.47B | 1.29B | 2.32B | 5.712 | 10.642 |
| 48q4kv | prefill | 8 | 0.49B | 1.49B | 2.34B | 5.632 | 10.692 |
| 48q4kv | prefill | 32 | 0.50B | 1.48B | 2.43B | 5.562 | 10.552 |
| 48q4kv | decode | 1 | 0.59B | 0.67B | 0.80B | 1.56B | 3.002 |
| 48q4kv | decode | 8 | 0.64B | 1.27B | 2.032 | 7.052 | 13.482 |
| 48q4kv | decode | 32 | 1.15B | 2.842 | 5.422 | 19.962 | 37.912 |
| 64q8kv | prefill | 1 | 0.34B | 1.04B | 1.64B | 3.812 | 7.362 |
| 64q8kv | prefill | 8 | 0.36B | 1.04B | 1.70B | 3.932 | 7.542 |
| 64q8kv | prefill | 32 | 0.38B | 1.10B | 1.75B | 4.022 | 7.532 |
| 64q8kv | decode | 1 | 0.67B | 0.80B | 1.08B | 2.502 | 4.862 |
| 64q8kv | decode | 8 | 0.86T | 1.862 | 3.372 | 12.212 | 23.342 |
| 64q8kv | decode | 32 | 1.42B | 4.012 | 7.712 | 28.432 | 53.702 |
| 80q8kv | prefill | 1 | 0.42B | 1.28B | 2.10B | 4.752 | 9.032 |
| 80q8kv | prefill | 8 | 0.44B | 1.32B | 2.15B | 4.812 | 9.202 |
| 80q8kv | prefill | 32 | 0.45B | 1.34B | 2.18B | 4.892 | 9.202 |
| 80q8kv | decode | 1 | 0.68T | 0.80B | 1.08B | 2.402 | 4.292 |
| 80q8kv | decode | 8 | 0.94B | 1.822 | 3.262 | 11.642 | 21.782 |
| 80q8kv | decode | 32 | 1.39B | 3.922 | 7.522 | 26.942 | 52.332 |
| 64q4kv | prefill | 1 | 0.62B | 1.90B | 3.07B | 7.442 | 14.272 |
| 64q4kv | prefill | 8 | 0.65B | 2.02B | 3.10B | 7.432 | 14.192 |
| 64q4kv | prefill | 32 | 0.66B | 2.00B | 3.16B | 7.312 | 13.862 |
| 64q4kv | decode | 1 | 0.59T | 0.68B | 0.81B | 1.55B | 2.612 |
| 64q4kv | decode | 8 | 0.66B | 1.26B | 2.022 | 6.882 | 13.172 |
| 64q4kv | decode | 32 | 1.10B | 2.842 | 5.382 | 19.452 | 37.102 |

## 48q4kv prefill, batch 8

| Method / indexer | 16k | 64k | 128k | 512k | 1024k |
|---|---:|---:|---:|---:|---:|
| token identical/64 | 0.28 | 0.47 | 0.51 | 0.60 | 0.52 |
| token identical/128 | 0.25 | 0.39 | 0.42 | 0.46 | 0.42 |
| token reduced/64 | 0.35 | 0.65 | 0.67 | 0.67 | 0.40 |
| token reduced/128 | 0.33 | 0.61 | 0.62 | 0.61 | 0.38 |
| block identical/64 | 0.36 | 0.66 | 0.77 | 0.93 | 0.96 |
| block identical/128 | 0.31 | 0.53 | 0.59 | 0.67 | 0.68 |
| block reduced/64 | 0.49 | 1.49 | 2.34 | 4.54 | 5.44 |
| block reduced/128 | 0.47 | 1.27 | 1.82 | 2.93 | 3.28 |
| two_stage identical/64 | 0.20 | 0.68 | 1.27 | 4.44 | 8.06 |
| two_stage identical/128 | 0.15 | 0.50 | 0.91 | 3.14 | 5.71 |
| two_stage reduced/64 | 0.22 | 0.77 | 1.47 | 5.63 | 10.69 |
| two_stage reduced/128 | 0.17 | 0.61 | 1.14 | 4.14 | 7.76 |

## 48q4kv decode, batch 8

| Method / indexer | 16k | 64k | 128k | 512k | 1024k |
|---|---:|---:|---:|---:|---:|
| token identical/64 | 0.62 | 1.07 | 1.48 | 2.13 | 2.37 |
| token identical/128 | 0.55 | 0.89 | 1.09 | 1.38 | 1.48 |
| token reduced/64 | 0.63 | 1.03 | 1.26 | 1.57 | 1.60 |
| token reduced/128 | 0.59 | 0.89 | 1.04 | 1.21 | 1.23 |
| block identical/64 | 0.60 | 1.16 | 1.65 | 2.77 | 3.20 |
| block identical/128 | 0.54 | 0.90 | 1.15 | 1.56 | 1.68 |
| block reduced/64 | 0.64 | 1.27 | 1.83 | 3.23 | 3.82 |
| block reduced/128 | 0.60 | 1.06 | 1.40 | 2.03 | 2.23 |
| two_stage identical/64 | 0.50 | 1.15 | 2.01 | 7.01 | 13.48 |
| two_stage identical/128 | 0.46 | 1.05 | 1.81 | 6.24 | 11.92 |
| two_stage reduced/64 | 0.52 | 1.17 | 2.03 | 7.05 | 13.36 |
| two_stage reduced/128 | 0.49 | 1.10 | 1.90 | 6.55 | 12.26 |

## 64q8kv prefill, batch 8

| Method / indexer | 16k | 64k | 128k | 512k | 1024k |
|---|---:|---:|---:|---:|---:|
| token identical/64 | 0.21 | 0.35 | 0.39 | 0.39 | 0.43 |
| token identical/128 | 0.20 | 0.32 | 0.36 | 0.36 | 0.38 |
| token reduced/64 | 0.24 | 0.44 | 0.47 | 0.45 | 0.49 |
| token reduced/128 | 0.23 | 0.40 | 0.43 | 0.41 | 0.44 |
| block identical/64 | 0.28 | 0.54 | 0.67 | 0.81 | 0.85 |
| block identical/128 | 0.25 | 0.45 | 0.54 | 0.62 | 0.67 |
| block reduced/64 | 0.36 | 1.04 | 1.70 | 3.14 | 3.70 |
| block reduced/128 | 0.34 | 0.88 | 1.31 | 2.01 | 2.23 |
| two_stage identical/64 | 0.14 | 0.46 | 0.92 | 3.13 | 5.69 |
| two_stage identical/128 | 0.11 | 0.34 | 0.66 | 2.23 | 4.08 |
| two_stage reduced/64 | 0.16 | 0.53 | 1.05 | 3.93 | 7.54 |
| two_stage reduced/128 | 0.12 | 0.41 | 0.82 | 3.01 | 5.61 |

## 64q8kv decode, batch 8

| Method / indexer | 16k | 64k | 128k | 512k | 1024k |
|---|---:|---:|---:|---:|---:|
| token identical/64 | 0.77 | 1.45 | 1.92 | 2.63 | 2.13 |
| token identical/128 | 0.68 | 0.96 | 1.25 | 1.51 | 1.32 |
| token reduced/64 | 0.86 | 1.45 | 1.83 | 2.27 | 2.40 |
| token reduced/128 | 0.79 | 1.19 | 1.41 | 1.64 | 1.69 |
| block identical/64 | 0.77 | 1.56 | 2.15 | 3.19 | 3.48 |
| block identical/128 | 0.65 | 1.10 | 1.35 | 1.68 | 1.75 |
| block reduced/64 | 0.84 | 1.76 | 2.49 | 3.86 | 4.30 |
| block reduced/128 | 0.76 | 1.38 | 1.77 | 2.33 | 2.47 |
| two_stage identical/64 | 0.63 | 1.84 | 3.20 | 11.68 | 21.32 |
| two_stage identical/128 | 0.57 | 1.54 | 2.79 | 10.02 | 18.06 |
| two_stage reduced/64 | 0.70 | 1.86 | 3.37 | 12.21 | 23.34 |
| two_stage reduced/128 | 0.64 | 1.70 | 3.04 | 10.87 | 20.52 |

## 80q8kv prefill, batch 8

| Method / indexer | 16k | 64k | 128k | 512k | 1024k |
|---|---:|---:|---:|---:|---:|
| token identical/64 | 0.24 | 0.40 | 0.45 | 0.49 | 0.43 |
| token identical/128 | 0.22 | 0.34 | 0.37 | 0.38 | 0.33 |
| token reduced/64 | 0.30 | 0.56 | 0.59 | 0.56 | 0.60 |
| token reduced/128 | 0.29 | 0.52 | 0.55 | 0.51 | 0.55 |
| block identical/64 | 0.32 | 0.58 | 0.69 | 0.78 | 0.80 |
| block identical/128 | 0.26 | 0.43 | 0.49 | 0.52 | 0.58 |
| block reduced/64 | 0.44 | 1.32 | 2.15 | 3.87 | 4.57 |
| block reduced/128 | 0.41 | 1.12 | 1.65 | 2.48 | 2.76 |
| two_stage identical/64 | 0.17 | 0.59 | 1.14 | 3.78 | 6.79 |
| two_stage identical/128 | 0.13 | 0.43 | 0.82 | 2.67 | 4.84 |
| two_stage reduced/64 | 0.19 | 0.67 | 1.32 | 4.81 | 9.20 |
| two_stage reduced/128 | 0.15 | 0.53 | 1.04 | 3.70 | 6.88 |

## 80q8kv decode, batch 8

| Method / indexer | 16k | 64k | 128k | 512k | 1024k |
|---|---:|---:|---:|---:|---:|
| token identical/64 | 0.85 | 1.41 | 1.86 | 2.62 | 2.70 |
| token identical/128 | 0.72 | 1.05 | 1.27 | 1.57 | 1.65 |
| token reduced/64 | 0.92 | 1.43 | 1.80 | 2.24 | 2.37 |
| token reduced/128 | 0.84 | 1.18 | 1.39 | 1.62 | 1.68 |
| block identical/64 | 0.84 | 1.51 | 2.08 | 3.14 | 3.46 |
| block identical/128 | 0.70 | 1.07 | 1.32 | 1.68 | 1.77 |
| block reduced/64 | 0.94 | 1.73 | 2.42 | 3.80 | 4.20 |
| block reduced/128 | 0.85 | 1.38 | 1.74 | 2.31 | 2.45 |
| two_stage identical/64 | 0.73 | 1.69 | 3.05 | 10.99 | 21.25 |
| two_stage identical/128 | 0.62 | 1.45 | 2.60 | 9.39 | 17.63 |
| two_stage reduced/64 | 0.80 | 1.82 | 3.26 | 11.64 | 21.78 |
| two_stage reduced/128 | 0.73 | 1.65 | 2.94 | 10.50 | 19.90 |

## 64q4kv prefill, batch 8

| Method / indexer | 16k | 64k | 128k | 512k | 1024k |
|---|---:|---:|---:|---:|---:|
| token identical/64 | 0.37 | 0.64 | 0.69 | 0.80 | 0.71 |
| token identical/128 | 0.32 | 0.52 | 0.54 | 0.60 | 0.54 |
| token reduced/64 | 0.46 | 0.89 | 0.90 | 0.89 | 0.53 |
| token reduced/128 | 0.44 | 0.82 | 0.83 | 0.81 | 0.50 |
| block identical/64 | 0.47 | 0.91 | 1.03 | 1.21 | 1.25 |
| block identical/128 | 0.40 | 0.68 | 0.74 | 0.83 | 0.85 |
| block reduced/64 | 0.65 | 2.02 | 3.10 | 5.99 | 7.16 |
| block reduced/128 | 0.62 | 1.73 | 2.45 | 3.89 | 4.39 |
| two_stage identical/64 | 0.26 | 0.92 | 1.69 | 5.77 | 10.56 |
| two_stage identical/128 | 0.20 | 0.67 | 1.20 | 4.05 | 7.43 |
| two_stage reduced/64 | 0.30 | 1.06 | 1.96 | 7.43 | 14.19 |
| two_stage reduced/128 | 0.23 | 0.83 | 1.53 | 5.49 | 10.33 |

## 64q4kv decode, batch 8

| Method / indexer | 16k | 64k | 128k | 512k | 1024k |
|---|---:|---:|---:|---:|---:|
| token identical/64 | 0.63 | 1.04 | 1.46 | 2.13 | 2.40 |
| token identical/128 | 0.56 | 0.84 | 1.05 | 1.33 | 1.42 |
| token reduced/64 | 0.65 | 0.99 | 1.27 | 1.59 | 1.62 |
| token reduced/128 | 0.62 | 0.87 | 1.03 | 1.24 | 1.24 |
| block identical/64 | 0.61 | 1.14 | 1.62 | 2.78 | 3.21 |
| block identical/128 | 0.55 | 0.89 | 1.14 | 1.56 | 1.68 |
| block reduced/64 | 0.66 | 1.26 | 1.82 | 3.26 | 3.86 |
| block reduced/128 | 0.63 | 1.06 | 1.41 | 2.06 | 2.26 |
| two_stage identical/64 | 0.52 | 1.13 | 1.97 | 6.80 | 13.07 |
| two_stage identical/128 | 0.47 | 1.02 | 1.77 | 6.08 | 11.47 |
| two_stage reduced/64 | 0.54 | 1.16 | 2.02 | 6.88 | 13.17 |
| two_stage reduced/128 | 0.51 | 1.10 | 1.90 | 6.42 | 12.11 |

## Stage breakdowns

**48q4kv, prefill, 128k, batch 8** (stage GPU ms)

| Method / indexer | Wall | main_projection | index_projection | block_pool | score | coarse_score_select | fine_score | select | expand | sparse_attention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| token identical/64 | 21.077 | 0.414 | 0.177 | - | 11.871 | - | - | 5.127 | 0.446 | 3.097 |
| token reduced/64 | 16.026 | 0.411 | 0.060 | - | 2.115 | - | - | 10.217 | 0.443 | 2.877 |
| block identical/64 | 13.993 | 0.408 | 0.178 | - | 10.845 | - | - | 0.356 | 0.305 | 1.878 |
| block reduced/64 | 4.621 | 0.412 | 0.061 | - | 1.579 | - | - | 0.458 | 0.302 | 1.783 |
| two_stage identical/64 | 8.536 | 0.414 | 0.177 | 0.006 | - | 0.524 | 3.448 | 1.336 | 0.610 | 1.954 |
| two_stage reduced/64 | 7.356 | 0.412 | 0.064 | 0.006 | - | 0.529 | 2.085 | 1.712 | 0.609 | 1.877 |

**48q4kv, decode, 128k, batch 8** (stage GPU ms)

| Method / indexer | Wall | main_projection | index_projection | block_pool | score | coarse_score_select | fine_score | select | expand | sparse_attention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| token identical/64 | 0.375 | 0.067 | 0.040 | - | 0.141 | - | - | 0.081 | 0.021 | 0.050 |
| token reduced/64 | 0.439 | 0.067 | 0.032 | - | 0.111 | - | - | 0.185 | 0.019 | 0.050 |
| block identical/64 | 0.335 | 0.065 | 0.040 | - | 0.135 | - | - | 0.025 | 0.034 | 0.051 |
| block reduced/64 | 0.302 | 0.066 | 0.031 | - | 0.110 | - | - | 0.028 | 0.034 | 0.050 |
| two_stage identical/64 | 0.275 | 0.069 | 0.041 | 0.007 | - | 0.039 | 0.026 | 0.033 | 0.042 | 0.050 |
| two_stage reduced/64 | 0.273 | 0.066 | 0.033 | 0.007 | - | 0.040 | 0.020 | 0.043 | 0.040 | 0.050 |

**64q8kv, prefill, 512k, batch 8** (stage GPU ms)

| Method / indexer | Wall | main_projection | index_projection | block_pool | score | coarse_score_select | fine_score | select | expand | sparse_attention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| token identical/64 | 149.882 | 0.564 | 0.225 | - | 78.562 | - | - | 55.794 | 2.739 | 12.917 |
| token reduced/64 | 129.162 | 0.564 | 0.062 | - | 17.191 | - | - | 98.305 | 2.742 | 11.496 |
| block identical/64 | 71.724 | 0.564 | 0.223 | - | 65.185 | - | - | 0.743 | 0.573 | 4.452 |
| block reduced/64 | 18.529 | 0.563 | 0.061 | - | 12.515 | - | - | 1.189 | 0.572 | 3.594 |
| two_stage identical/64 | 18.623 | 0.565 | 0.223 | 0.006 | - | 1.541 | 7.885 | 2.662 | 1.213 | 4.525 |
| two_stage reduced/64 | 14.806 | 0.564 | 0.063 | 0.006 | - | 1.494 | 4.174 | 3.425 | 1.212 | 3.848 |
