#!/usr/bin/env bash
# T2 门之后的接力脚本（**只在门判为"通过"之后由人/我手动启动**，本脚本自己不判门）。
#
# 为什么单独做一个文件：实验 B（6 run）和实验 C（8 run）都要占满 4 张卡，
# 不能并行；串行接力可以把"门通过"到"C 出数"之间的人工等待压到零，
# 同时保证**门本身不被脚本绕过**——判门是 scripts/v3_expA_v2_gate.py 的事，
# 拉训练是这里的事，两件事分开，就不会出现"训练已经跑了、门被默认通过"的情况。
#
# 用法：
#   nohup bash runs/v3_chain_after_T2.sh > runs/v3_chain_after_T2.log 2>&1 &
set -u
cd /ssd1/wjb/SVD-VALID || exit 1

echo "T2_CHAIN_START $(date -Is)"
bash runs/v3_expB_200k/launch.sh >> runs/v3_chain_after_T2.log 2>&1
echo "EXP_B_EXIT rc=$? $(date -Is)"
bash runs/v3_expC_200k/launch.sh >> runs/v3_chain_after_T2.log 2>&1
echo "EXP_C_EXIT rc=$? $(date -Is)"
echo "T2_CHAIN_DONE $(date -Is)"
