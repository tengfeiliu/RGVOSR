# RL-SR 阶段 A–E 运行手册

这套实现把一次 SR 输出视为外层动作。第 1 轮仍使用冻结的 F0；第 2 轮及以后使用共享精修策略 `G_phi(x, y_k, k+1)`。其中 `x` 是原始 LR，`y_k` 是当前 SR 输出。FLUX.2 Transformer、VAE、文本编码器、F0 和 MoE router 均不会在 C/E 中更新。

需要的既有数据只有配对 LR/HR JSONL，以及已有 F0 adapter。当前分支的 `datasets/LSDIR_precrop512/train.iqa_caption_suggestion.jsonl` 已经满足配对训练需要，因此不需要新增人工标注。自动 reward_v1 需要可用的 PyIQA；没有 PyIQA 时可使用 `--no_quality` 做联调，但这不能作为正式 RL 训练奖励。

运行时的文件路径可以因服务器而不同。状态、rollout 和 adapter 的哈希只包含数据内容、根目录下的相对 key、采样器配置和模型张量，不包含绝对路径、时间、GPU 或输出目录。

## A：固定策略与配置

从现有 F0 checkpoint 准备 `rg_flux_adapters` 目录。新配置为 [rl_sr_refinement_flux2_klein_moe.yaml](../configs/rl_sr_refinement_flux2_klein_moe.yaml)，已按 checkpoint-00036000 固化 `iqa_suggestion + caption`、condition8 MoE、rank 32、4 experts、`uniform` flow-matching sigma。只需把其中 `model.flux_model_path` 改成当前服务器的模型位置；这不会影响任何 artifact hash。命令中的 `<F0_ADAPTER>` 可传 checkpoint 根目录或其下 `rg_flux_adapters`；策略哈希会自动只计算后者，不会包含 optimizer/training state。

不要把 `training.freeze_flux_transformer` 改为 `false`。C 和 E 的入口会在创建 optimizer 前检查此项，并将未列入 LoRA/condition adapter 白名单的参数全部冻结。

## B：生成第一轮状态及后续多轮状态

先仅生成 F0 的第 1 轮输出 `y_1`。一键入口会从配置的
`data.jsonl_path` 自动生成输入清单，因此不需要单独提供 `LQ_ROOT` 或
另一份 `PAIRED_JSONL`。下面仅保留手工调试示例；其中 `<DATA_JSONL>` 就是
配置中的 `data.jsonl_path`。

```powershell
python tools/run_rg_flux_iterative_inference.py `
  --checkpoint <F0_ADAPTER> `
  --config configs/rl_sr_refinement_flux2_klein_moe.yaml `
  --input <LQ_INPUT_LIST> --jsonl_path <DATA_JSONL> `
  --output_dir <RUN_ROOT>/f0_round1 --iterations 1 --upscale 1
```

把 lineage 转换为 C 所需状态。`--lr_root` 和 `--hr_root` 现在是可选项：
不提供时，工具自动使用内容哈希构造相对 key，因此数据切换到另一台服务器或
挂载目录时，state ID 和 artifact hash 保持不变。

```powershell
python tools/build_sr_refinement_states.py `
  --lineage_jsonl <RUN_ROOT>/f0_round1/sample_lineage.jsonl `
  --source_jsonl <DATA_JSONL> --dataset_id lsdir_precrop512_v1 `
  --artifact_root <RUN_ROOT>/f0_round1 `
  --producer_adapter <F0_ADAPTER> --output_jsonl <RUN_ROOT>/states_for_c.jsonl --max_round 2
```

状态构建会把原始 prompt 和冻结 condition8 router 输入写入 metadata，并把它们的哈希写入 `state_id`。如果原始 JSONL 中缺失 `hq_path`、`lq_path`、IQA/suggestion profile 或生成的 `y_1`，该命令会立即失败并指出样本；不需要人工补标，只需修复已有配对或推理输出。

## C：共享多轮 SFT

```powershell
python train_rg_flux_refiner_sft.py `
  --config configs/rl_sr_refinement_flux2_klein_moe.yaml `
  --state_jsonl <RUN_ROOT>/states_for_c.jsonl `
  --init_adapter <F0_ADAPTER> --output_dir <RUN_ROOT>/g_sft
```

输出 `<RUN_ROOT>/g_sft/rg_flux_adapters` 同时是 `G_ref`、第一轮 `G_old` 和初始 `G_phi`。训练目标为 `G_phi(x, y_1, 2) -> HR`。原始 LR 的 `SRRefinementConditionAdapter` 使用零门控初始化，因此它刚接入时严格等价于原输入模型；训练后才逐步利用 `x` 和轮次编码修正漂移。

用 C 的策略重新生成 1–4 轮。第 1 轮固定 F0，第 2 轮起自动载入 G_SFT，并将每张原始 LR 作为 anchor。这个命令没有两轮上限。

```powershell
python tools/run_rg_flux_iterative_inference.py `
  --checkpoint <F0_ADAPTER> --refiner_checkpoint <RUN_ROOT>/g_sft/rg_flux_adapters `
  --config configs/rl_sr_refinement_flux2_klein_moe.yaml `
  --input <LQ_INPUT_LIST> --jsonl_path <DATA_JSONL> `
  --output_dir <RUN_ROOT>/sft_rounds --iterations 4 --upscale 1

python tools/build_sr_refinement_states.py `
  --lineage_jsonl <RUN_ROOT>/sft_rounds/sample_lineage.jsonl `
  --source_jsonl <DATA_JSONL> --dataset_id lsdir_precrop512_v1 `
  --artifact_root <RUN_ROOT>/sft_rounds `
  --producer_adapter <RUN_ROOT>/g_sft/rg_flux_adapters `
  --round_producer_adapter 1=<F0_ADAPTER> `
  --output_jsonl <RUN_ROOT>/states_for_rl.jsonl --max_round 4
```

## D：自动 reward_v1

先从冻结 `G_old` 为每个状态采样 K 个完整动作。保存的 `z0.pt` 是 rectified-flow ODE 结束时的原始 latent；PNG 只用于 reward 查看，不能替换该 latent。

```powershell
python tools/collect_sr_output_rollouts.py `
  --config configs/rl_sr_refinement_flux2_klein_moe.yaml `
  --state_jsonl <RUN_ROOT>/states_for_rl.jsonl `
  --old_adapter <RUN_ROOT>/g_sft/rg_flux_adapters `
  --output_dir <RUN_ROOT>/rollouts_gsft --num_candidates 4

python tools/calibrate_sr_reward.py `
  --rollout_jsonl <RUN_ROOT>/rollouts_gsft/rollouts.jsonl `
  --output_json <RUN_ROOT>/reward_calibration.json --device cuda

python tools/score_sr_rollouts.py `
  --state_jsonl <RUN_ROOT>/states_for_rl.jsonl `
  --rollout_jsonl <RUN_ROOT>/rollouts_gsft/rollouts.jsonl `
  --calibration_json <RUN_ROOT>/reward_calibration.json `
  --output_jsonl <RUN_ROOT>/scored_rollouts.jsonl --device cuda
```

`reward_v1` 的质量项使用多个 NR-IQA 指标经分位数校准后取稳健中位数增益。忠实度项将候选和父输出下采样到原始 LR 坐标，比较低频、边缘和颜色一致性。局部项从 `y_k` 与上采样 `x` 的残差和边缘得到风险区域，奖励该区域的细节增益，并惩罚其余区域的不必要改动。违规项检测源低频偏离、色偏、过强高频/振铃和大面积改动；超过硬阈值的候选获得 `p=0`。置信度结合 NR-IQA 是否同意、数值有效性和违规程度；低置信度候选的偏好概率会收缩到 0.5。

以上过程不使用匿名人工比较。`reward_calibration.json` 只保存指标的 5%/95% 分位数和方向，不保存路径或样本图像。它是自动标尺，不等同于人的偏好校准；后续若训练 reward_v2/VLM Judge，才需要追加少量人工比较数据。

## E：输出级 DiffusionNFT

```powershell
python train_rg_flux_output_rl.py `
  --config configs/rl_sr_refinement_flux2_klein_moe.yaml `
  --state_jsonl <RUN_ROOT>/states_for_rl.jsonl `
  --scored_rollout_jsonl <RUN_ROOT>/scored_rollouts.jsonl `
  --reference_adapter <RUN_ROOT>/g_sft/rg_flux_adapters `
  --old_adapter <RUN_ROOT>/g_sft/rg_flux_adapters `
  --policy_adapter <RUN_ROOT>/g_sft/rg_flux_adapters `
  --output_dir <RUN_ROOT>/g_rl_01
```

E 会先检查 scored buffer 的 `policy_adapter_sha256` 是否与 `G_old` 一致，防止把旧动作用在新策略上。对每个保存的 `z0` 重新采样 `tau, epsilon`，构造 `z_tau=(1-tau)z0+tau*epsilon` 和原生 velocity 目标 `epsilon-z0`。冻结 `G_old` 产生 `v_old`，冻结 `G_ref` 提供保守约束；当前 `G_phi` 通过

```text
p || (1-beta) v_old + beta v_phi - v_target ||²
+ (1-p) || (1+beta) v_old - beta v_phi - v_target ||²
```

接收正、负两侧的输出奖励信号。SFT replay 同时持续用 HR 锚定 G_phi，避免只追逐 reward 而损坏已学到的保真能力。

每完成一次 E 更新，必须把新输出 adapter 作为下一轮 `G_old` 重新收集 rollout、校准/评分、再更新。不要对同一个 buffer 连续做多轮更新；这样 `G_old`、动作 latent 和奖励的对应关系始终正确。

## 单卡后台运行

仓库提供 [run_rl_sr_stage_ae.sh](../tools/run_rl_sr_stage_ae.sh)。它会依次完成：

1. 从 `config.data.jsonl_path` 自动取出 LQ，并生成运行时输入清单；
2. 生成 F0 的训练状态、训练共享 SFT、生成多轮训练状态；
3. 自动采样、校准和计算 reward，再执行输出级 NFT；
4. 对 `G_SFT` 和最终 `G_RL` 自动在 `RealLQ259`、`RealLR200` 进行 1–`iterations` 轮推理与指标评估。

它不需要 `LQ_ROOT`、`DATASET_ROOT` 或 `PAIRED_JSONL`。训练仍读取
`data.jsonl_path`；评估读取原始的 `rl_sr.evaluation.jsonl_path`
（默认 `datasets/inference.iqa_caption_suggestion.jsonl`）。该 JSONL 不会被
拆分、修改或复制为条件文件：脚本只按 `dataset_filter` 生成运行时 LQ 输入列表，
再把原始 JSONL 直接传给推理器。配置中的 `expected_count: 259/200` 会在开始推理前
验证两套数据是否完整。
每次启动会创建唯一目录，名称包含 F0 step、FLUX 配置、prompt、condition、轮数、
候选数、训练步数和时间，例如
`rlsr_f0-036000_flux2-klein_flux2-image-concat_iqa-suggestion_ccondition8_s512_r4_k4_sft1000_nft250_eval-reallq259-reallr200_260912-101530`。

评估 JSONL 每条记录必须有可访问的 `lq_path`，以及与当前 `iqa_suggestion + caption`
设置匹配的 `unipercept_raw.profile`。记录还应以 `dataset` 或 `dataset_name` 标记为
`RealLQ259`、`RealLR200`，供脚本精确分组。当前评估是 NR-IQA，因此不需要 HR、
`HQ_ROOT` 或额外的评估根目录；如果以后加入 PSNR/SSIM 等全参考指标，才需要在 JSONL
中保留可访问的 `hq_path` 并扩展评估器。

```bash
cd <REPO_ROOT>
chmod +x tools/run_rl_sr_stage_ae.sh

export F0_CHECKPOINT=<CHECKPOINT-00036000_OR_RG_FLUX_ADAPTERS>
export CONFIG=<REPO_ROOT>/configs/rl_sr_refinement_flux2_klein_moe.yaml

nohup env CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
  CONDA_ENV=sr-flux2 \
  bash tools/run_rl_sr_stage_ae.sh all \
  > "rlsr_launcher_$(date +%y%m%d-%H%M%S).log" 2>&1 < /dev/null &
```

启动日志的第一行会打印自动创建的 `RL-SR run directory`。详细日志保存在该目录的
`logs/`；评估趋势在 `06_sft_multiround_evaluation/metric_trends.csv` 和
`11_rl_multiround_evaluation/metric_trends.csv`。每轮目录保持现有格式：
`round_01/RealLQ259/`、`round_01/RealLR200/`，其下的 `metrics/` 保存逐图和汇总
NR-IQA。脚本还会生成
`12_sft_to_rl_evaluation.json`，将 G_RL 相对 G_SFT 的每轮每项指标变化统一成
“正数=变好”的 `oriented_delta`。输出目录的根路径由 `rl_sr.output_root` 控制，默认是仓库下的 `exp_rg_flux_rl/`，可用
`RL_SR_OUTPUT_ROOT` 临时覆盖而不影响 hash。

若任务中断，设置已有运行目录后只重跑未完成阶段，避免覆盖已生成的 inference：

```bash
RL_SR_RUN_DIR=<自动创建的运行目录> \
  F0_CHECKPOINT=<CHECKPOINT-00036000_OR_RG_FLUX_ADAPTERS> \
  CONDA_ENV=sr-flux2 bash tools/run_rl_sr_stage_ae.sh eval
```

可把 `eval` 替换成 `f0`、`c`、`multiround`、`reward` 或 `e`。这些单阶段命令仍会
自动恢复 JSONL 输入清单；不会要求数据根目录参数。
