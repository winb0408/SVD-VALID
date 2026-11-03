# Round9 对比图索引

split `primary`，场景 e902, e903, e904, e905。

显示：线性 PD 的 dB（`10log10(pd / peak_ref)`），**每行一个标尺**，标尺由该场景**真值的 p99** 定（与任何方法无关，不逐列拉伸）。裁剪与帧号对同一行所有列相同。

| 场景 | 帧 z | 真值 p99（dB 参考） | 裁剪 [y0:y1, x0:x1] | 图 |
|---|---|---|---|---|
| e902 | 7 | 11.991 | [0, 64, 0, 64] | `reports/v3_round9/figs/s0_e902_compare.png` |
| e903 | 24 | 10.181 | [0, 64, 0, 64] | `reports/v3_round9/figs/s0_e903_compare.png` |
| e904 | 9 | 11.538 | [0, 64, 0, 64] | `reports/v3_round9/figs/s0_e904_compare.png` |
| e905 | 22 | 13.173 | [0, 64, 0, 64] | `reports/v3_round9/figs/s0_e905_compare.png` |
