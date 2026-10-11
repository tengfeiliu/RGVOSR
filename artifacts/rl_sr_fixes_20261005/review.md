# RL-SR修复代码审查记录

2026-10-05，在原RL-SR分支由独立审查agent检查本轮实现。审查时固定路由、FP32 NFT、空间锚点、图像loss、多轮数据与batch已实现，尚未进行最终GPU训练步验证。

无critical发现；3项重要发现均修复并有red_review.log / red_round_control.log → green_review.log证据：

| 发现 | 修复 | 回归 |
|---|---|---|
| round数量超过batch，起始轮次反复重置导致晚轮次漏采 | batch与epoch的连续轮次游标 | 8轮/4batch跨batch覆盖；10轮/4batch跨epoch覆盖 |
| metadata声明refinement，但condition_adapters.pt不存在时绕过加载验收 | 在读取metadata时检查文件；加载时检查全部新增权重 | 文件缺失/anchor_query参数缺失均拒绝 |
| 全局iterations>=3把bootstrap条件套给已有C缓存和推理 | bootstrap有独立sft_bootstrap_rounds>=2；iterations>=1 | 已有C缓存、iterations=2仍可进入训练 |

随后真实GPU训练发现两项额外问题，按复现→修复→重跑处理：

- topk路由在BF16 CUDA autocast下softmax返回FP32，与BF16 scatter目标冲突；改为采用实际softmax dtype，red_amp.log → CUDA回归通过。
- E policy与replay两个梯度图同时保留导致显存分配失败；顺序反向、最后只更新一次。red_memory.log → 两种累积缩放测试通过；真实online C/E batch4成功，峰值66.4/66.9GiB。

最终相关测试62通过；全套299通过/与基线一致的7失败，详细说明在docs/rl_sr_branch_fixes_20261005.md。独立审查未覆盖之后新增的两项GPU修复；这些由真实训练与回归测试确认。正式SR质量、长期收敛和多轮收益尚待独立实验。
