# 残差 proxy 的标定：这把尺子到底量的是什么

数据：`metric_calibration_*.json`（overlap 扫描，真实折、真实秩规则、真实匹配算子）。

## SA64

- **锚点 1（overlap=1 必须精确为 0）**：64 个格点，最大 |rel_matched| = 0.00e+00 ✅

- **锚点 2（overlap=0 复现审计）**：按 (variant, rule) 汇总的折均值 rel_matched ——

  | variant | rule | 折数 | 均值 | 中位 | 最小 | 最大 |
  |---|---|---|---|---|---|---|
  | as_impl | clutter_knee | 16 | -0.305 | -0.283 | -0.559 | -0.059 |
  | as_impl | proportional | 16 | -0.259 | -0.217 | -0.493 | -0.025 |
  | f0_free | clutter_knee | 16 | -0.195 | -0.186 | -0.489 | -0.048 |
  | f0_free | proportional | 16 | -0.162 | -0.147 | -0.463 | -0.032 |

### rel_matched 随臂重叠的走势（折均值）

| variant | rule | split | N_A | ov=0 | ov=0.25 | ov=0.5 | ov=0.75 | ov=0.9 | ov=1 |
|---|---|---|---|---|---|---|---|---|---|
| as_impl | clutter_knee | sa64_cont3 | 133 | -0.326 | -0.234 | -0.174 | -0.079 | -0.031 | +0.000 |
| as_impl | clutter_knee | sa64_int3 | 133 | -0.256 | -0.032 | -0.024 | -0.016 | -0.005 | +0.000 |
| as_impl | clutter_knee | sa64_int5 | 80 | -0.313 | -0.138 | -0.104 | -0.069 | -0.036 | +0.000 |
| as_impl | proportional | sa64_cont3 | 133 | -0.308 | -0.205 | -0.145 | -0.055 | -0.018 | +0.000 |
| as_impl | proportional | sa64_int3 | 133 | -0.206 | -0.014 | -0.011 | -0.006 | -0.001 | +0.000 |
| as_impl | proportional | sa64_int5 | 80 | -0.260 | -0.103 | -0.077 | -0.053 | -0.029 | +0.000 |
| f0_free | clutter_knee | sa64_cont3 | 132 | -0.330 | -0.246 | -0.186 | -0.087 | -0.038 | +0.000 |
| f0_free | clutter_knee | sa64_int3 | 132 | -0.065 | -0.045 | -0.033 | -0.021 | -0.005 | +0.000 |
| f0_free | clutter_knee | sa64_int5 | 79 | -0.193 | -0.136 | -0.100 | -0.064 | -0.032 | +0.000 |
| f0_free | proportional | sa64_cont3 | 132 | -0.317 | -0.222 | -0.159 | -0.062 | -0.023 | +0.000 |
| f0_free | proportional | sa64_int3 | 132 | -0.040 | -0.025 | -0.019 | -0.010 | -0.001 | +0.000 |
| f0_free | proportional | sa64_int5 | 79 | -0.153 | -0.107 | -0.077 | -0.050 | -0.025 | +0.000 |

### 门槛 `rel <= -0.10` 实际在测什么

最大重叠比例仍能过门槛者（即门槛能容忍两臂共享多少帧）：

| variant | rule | split | 通过的 overlap 上界 | 说明 |
|---|---|---|---|---|
| as_impl | clutter_knee | sa64_cont3 | <= 0.5 | 几乎任何非完全相同的一对都能过 |
| as_impl | clutter_knee | sa64_int3 | <= 0 |  |
| as_impl | clutter_knee | sa64_int5 | <= 0.5 | 几乎任何非完全相同的一对都能过 |
| as_impl | proportional | sa64_cont3 | <= 0.5 | 几乎任何非完全相同的一对都能过 |
| as_impl | proportional | sa64_int3 | <= 0 |  |
| as_impl | proportional | sa64_int5 | <= 0.25 |  |
| f0_free | clutter_knee | sa64_cont3 | <= 0.5 | 几乎任何非完全相同的一对都能过 |
| f0_free | clutter_knee | sa64_int3 | 无 |  |
| f0_free | clutter_knee | sa64_int5 | <= 0.5 | 几乎任何非完全相同的一对都能过 |
| f0_free | proportional | sa64_cont3 | <= 0.5 | 几乎任何非完全相同的一对都能过 |
| f0_free | proportional | sa64_int3 | 无 |  |
| f0_free | proportional | sa64_int5 | <= 0.25 |  |

### rel_matched 到底跟着什么走

- 与真实臂间噪声相关 rho_eff 的 Spearman：**+0.703**
- 与两臂功率图相似度 power_map_r 的 Spearman：**+0.976**
- 与 rho_eff 的 Pearson：+0.693；与 (1 - power_map_r) 的 Pearson：-0.932

- 按 (variant, rule, split) 分组看，随 overlap 单调不降的组：12/12（说明在**固定设计**内，尺子对'两臂共享多少帧'是单调的）

## HSA64

- **锚点 1（overlap=1 必须精确为 0）**：12 个格点，最大 |rel_matched| = 0.00e+00 ✅

- **锚点 2（overlap=0 复现审计）**：按 (variant, rule) 汇总的折均值 rel_matched ——

  | variant | rule | 折数 | 均值 | 中位 | 最小 | 最大 |
  |---|---|---|---|---|---|---|
  | as_impl | clutter_knee | 3 | -0.022 | -0.022 | -0.024 | -0.020 |
  | as_impl | proportional | 3 | -0.012 | -0.012 | -0.016 | -0.009 |
  | f0_free | clutter_knee | 3 | -0.021 | -0.022 | -0.022 | -0.020 |
  | f0_free | proportional | 3 | -0.012 | -0.012 | -0.014 | -0.009 |

### rel_matched 随臂重叠的走势（折均值）

| variant | rule | split | N_A | ov=0 | ov=0.25 | ov=0.5 | ov=0.75 | ov=0.9 | ov=1 |
|---|---|---|---|---|---|---|---|---|---|
| as_impl | clutter_knee | hsa64_int3 | 133 | -0.022 | -0.016 | -0.011 | -0.008 | -0.004 | +0.000 |
| as_impl | proportional | hsa64_int3 | 133 | -0.012 | -0.009 | -0.009 | -0.006 | -0.002 | +0.000 |
| f0_free | clutter_knee | hsa64_int3 | 132 | -0.021 | -0.016 | -0.011 | -0.008 | -0.004 | +0.000 |
| f0_free | proportional | hsa64_int3 | 132 | -0.012 | -0.010 | -0.009 | -0.006 | -0.002 | +0.000 |

### 门槛 `rel <= -0.10` 实际在测什么

最大重叠比例仍能过门槛者（即门槛能容忍两臂共享多少帧）：

| variant | rule | split | 通过的 overlap 上界 | 说明 |
|---|---|---|---|---|
| as_impl | clutter_knee | hsa64_int3 | 无 |  |
| as_impl | proportional | hsa64_int3 | 无 |  |
| f0_free | clutter_knee | hsa64_int3 | 无 |  |
| f0_free | proportional | hsa64_int3 | 无 |  |

### rel_matched 到底跟着什么走

- 与真实臂间噪声相关 rho_eff 的 Spearman：**+0.827**
- 与两臂功率图相似度 power_map_r 的 Spearman：**+0.928**
- 与 rho_eff 的 Pearson：+0.779；与 (1 - power_map_r) 的 Pearson：-0.939

- 按 (variant, rule, split) 分组看，随 overlap 单调不降的组：4/4（说明在**固定设计**内，尺子对'两臂共享多少帧'是单调的）

