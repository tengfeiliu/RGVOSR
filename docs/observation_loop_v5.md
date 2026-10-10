# V5：原始观测约束下的共享递归 SR 验证

本版本是在 V4 的失败结果之后新增的独立实验，工作树为 `/media/xd/data/code/RGVOSR-self-refine-v5`，Git 分支为 `RL-SR-observation-loop-v5`。旧版实验和结果保留。本文件描述已实现的方法；是否有效以新验证/测试结果为准。

## 1. 问题和可检验假设

此前后轮把上一轮生成图作为主要图像依据，原始 LR 通过 64 个池化 anchor 加到文本条件，且新增 gate 初始为零。传入 LR 不等于有效利用 LR。重复执行完整生成还可能重新制造纹理。因此分开检验：

1. 将 LR 作为有空间位置的图像条件，是否改善实际输出？
2. 将后轮改为有界的小幅修正，是否减少重复生成导致的漂移？
3. 轮次条件、跨轮隐藏状态和自身轨迹刷新，是否分别改善第 3、4 轮？

## 2. 保持的基础模型

- 基础模型：`/media/xd/data/models/FLUX.2-klein-base-4B`，BF16 主干；没有换成 FLUX.1。
- 固定首轮 F0：用户上传的 `checkpoint-00036000/rg_flux_adapters`，内容 SHA256 为 `bf8539cfd27dc5f43d9a202b6f68ab87cc2ae8c603ca37df2e0b19f909452fe0`。
- 首轮保留原有个性化 prompt 和 condition8 MoE。
- 后续 FLUX 对照使用通用 prompt，FP32 Qwen 编码后存为 BF16 `[1,512,7680]`；原始 condition8 仍描述原始 LR，原 MoE/router 全部冻结。
- 保留图像角色：前轮 SR 是原生 image concat 条件；原始 LR 是参考观测。
- 未更改 CUDA、驱动、Python 环境。

## 3. 两类实验不可混为一谈

### G：在 FLUX.2 上检验原始 LR 的输入位置

`G_generic` 是同一384张新heldout上的未经训练通用prompt多轮生成对照。`G_weak` 训练原有的 64 个文本侧 anchor；`G_spatial` 将 512 输入的 VAE latent 保留为 32×32、1024 个位置，通过归一化 Q/K/V cross-attention 注入 FLUX 的目标图像 token：

`h_target <- h_target + W_out Attention(Q(h_target), K(z_original_lr)+pos, V(z_original_lr))`

`W_out` 零初始化。只更新新增支路，前轮 SR 的 concat token 不改，F0 首轮绕过注入。初始输出严格等于原 F0，输出投影有非零梯度，真实优化后模型预测已改变。两组采用相同源图、轮次混合、200 次更新、有效 batch20 和相同训练目标。参数量分别记录，**不是严格等参数的容量消融**。

损失：flow matching MSE + 0.1 latent SmoothL1 + 0.1 解码图像 Charbonnier + 0.05 HR 梯度一致性；图像辅助损失采用对齐的 256 crop。评价时使用真实完整 25-NFE 采样，不以一步估计图像代替最终 SR。

另外在 64 张固定验证图上关闭 LR 分支、打乱 LR 图像，原始 condition8 和前轮 SR 保持一致。这能检验 LR 是否真正影响实际输出。

### C：独立的小型共享 visual Transformer 修正器

F0 负责第一轮，后面每轮执行一次同一个修正器：

`y_(k+1) = clamp(y_k + sigmoid(g_k) * 0.25*tanh(delta_k), -1, 1)`

每轮：前轮 RGB 和原始 LR 各经 stride16 的视觉编码器，512 图像产生 1024 个空间 token；width96、2 blocks、4 heads，包含图像 self-attention、LR cross-attention 和 MLP。轮次 embedding 与当前图像/LR 的全局特征组合为条件，调制 block；跨轮传递隐藏特征 `h_(k+1)`。RGB residual head 零初始化，初始保持输入，10% 训练样本采用 HR 作为近最优输入，学习保持。

**该修正器是新训练的小型 visual Transformer，不是对 FLUX 全主干直接加 loop；C 组没有文本或 MoE 输入。** 因而 C 组有效也只能先支持“在 FLUX 首轮之后共享迭代修正”的研究方向。

| 对照 | LR 通路 | 轮次条件 | 跨轮 state | 训练轨迹 |
|---|---|---|---|---|
| C0_weak | 池化为一个全局视觉 token | 关闭 | 关闭 | 固定旧轨迹，第2/3/4轮混合，单步 |
| C1_spatial | 1024空间 token | 关闭 | 关闭 | 与C0相同 |
| C2_round_state | 1024空间 token | 开启 | 关闭 | 与C0相同，各轮 embedding 均实际训练 |
| C3_recursive | 1024空间 token | 开启 | 开启 | 自己输出连续展开，训练1→2→3次修正 |
| C4_refresh | 1024空间 token | 开启 | 开启 | C3基础上周期刷新 old-policy 图像和对应 hidden |

C0→C1、C1→C2可比较单一输入机制。C2→C3同时改变展开训练和隐藏状态，应结合冻结模型的 `no_state` 消融判断，不能只凭跨组差值把收益全部归因于 hidden。

C 每组 1200 次更新、有效 batch256、512整图，学习率2e-4。全部实际输出对 HR 的 Charbonnier + 0.05 梯度一致性 + 0.1 avgpool4低频重建。没有 CLIPIQA、CLIPIQA+、MUSIQ、LPIPS 或 DISTS 训练损失；指标只用于验证和报告。

C4 每400次更新冻结一次当时的共享模型，生成每张训练图第2/3轮状态，保存连续 float32 RGB、对应 BF16 hidden、轮次、源图顺序和精确 producer checkpoint SHA。后续 batch 按50%比例混合这些 old 状态；new 通过梯度更新。没有使用 EMA，也没有加入 RL。展开时从正确的下一轮开始，不超过第4轮。

## 4. 数据规模和复用

- 训练：1024 个独立原始 HR 源图；复用已核验的 F0 首轮结果和旧轨迹作为训练输入，目标始终为配对 HR。
- 验证：128 个新的原始 HR 源图。
- 测试：256 个新的原始 HR 源图。
- 新 heldout 排除旧3000训练图的所有 HR 源组，以及此前 V2/V3/V4 已查看的64个 heldout 源组。训练、验证、测试之间按原始 HR 分组。
- LR 文件已上采样至512，来源为合成4倍退化；不能把 LR 的512文件尺寸误解为未退化。
- **没有证明新 heldout 独立于 F0 的36k预训练数据；本实验仅检验后续训练的泛化，不能作为论文最终无泄漏基准。**

实际独立的验证/测试照片共384张，每种最终模型都输出第1～4轮。共享F0和8个后轮模型对应9600张最终路径图像；中间checkpoint和关闭/打乱条件的图像另存。生成数量不能替代独立源图数量。

## 5. 判定方式

- RGB `[0,1]`、整张512、无裁边，配对 PSNR/SSIM/LPIPS/DISTS，独立每图记录。
- C 的6个checkpoint在固定64张验证图上跑实际第1～4轮，然后在完整128验证+256测试上报告选中模型。G 的候选在32验证图上比较第二轮，完整结果另检验递归表现。
- 验证候选通过门槛：R2相对F0 PSNR至少+0.02dB，SSIM不低于−0.002，LPIPS不高于+0.002，DISTS不高于+0.001；C还要求R4满足同样的F0门槛且PSNR不比R2下降超过0.02dB。
- G 的第二轮通过与完整递归通过单独记录，不能混称。
- 保留所有原始候选，即使门槛失败；不把回退 F0、HR oracle 选择或评价指标辅助选图计为递归提升。
- 逐图统计后轮相对首轮的增益、退化比例和2000次paired-bootstrap95%区间；同时检查固定源图的可视化，不只看均值。
- 门槛用于筛选后续研究候选，不是统计显著性或论文成功标准。

## 6. 运行保障与时间

A100 实测满512、batch256、3次真实修正展开及反向，首次合成输入测得38.12GiB/1.43秒；这是计算探针，不是完整训练平均速度。真实4图配对 smoke 的重建loss0.24597，输出head梯度norm0.006339，说明梯度链路可用，不说明SR效果成功。

真实 FLUX 输入测试：weak/spatial 初始输出均严格等于F0；一次更新后 velocity RMS 改变量分别0.00667、0.02093，梯度norm分别0.000736、0.0721，单图完整训练loss峰值约9.9GiB。VAE通过项目的自定义 `.to(device='cuda')` 迁移，保留原环境。

断点测试在真实 CUDA 上注入 update2 后、候选评估前的中断，恢复后完成候选2/4/6和old刷新2/4，图像链、hidden和来源校验通过。此8图测试仅验证恢复机制。

回归测试：新旧相关测试35项通过；另以实际子进程模拟supervisor重启，确认在旧child仍存活时不会启动竞争stage，随后正常完成prepare和report。

后台流程：prepare→prompt cache→新heldout F0→真实smoke→5组C训练和评测→未训练G对照→2组G训练和评测→报告。首轮已测batch8约62秒/批；按当前资源估计整个任务约12～22小时，GPU争用或OOM微batch回退会延长。

supervisor记录PID/start ticks/argv，重启先核对/等待尚存的自身child，stage加文件锁；每10秒heartbeat；OOM保留有效batch，降低microbatch后最多重试3次。模型、数据、配置和代码哈希固定。已提交的完整图像链不覆盖，坏缓存/来源不一致显式失败。失败写status和日志，不伪装完成。

一键启动/恢复：

```bash
bash /media/xd/data/code/RGVOSR-self-refine-v5/tools/start_sr_observation_loop_background.sh
```

运行目录：`exp_rg_flux_self_refine/observation_loop_v5_20261010/`。查看 `status.json`、`supervisor.log`、`logs/correctors.log`、`round_metrics_summary.csv`、`paired_bootstrap.json`、`report.md`、`inspection/` 和 `trajectories/`。
