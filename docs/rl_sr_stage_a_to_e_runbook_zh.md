# RL-SR 阶段 A–E 运行手册

这套实现把一次 SR 输出视为外层动作。第 1 轮仍使用冻结的 F0；第 2 轮及以后使用共享精修策略 `G_phi(x, y_k, k+1)`。其中 `x` 是原始 LR，`y_k` 是当前 SR 输出。FLUX.2 Transformer、VAE、文本编码器、F0 和 MoE router 均不会在 C/E 中更新。

需要的既有数据只有配对 LR/HR JSONL，以及已有 F0 adapter。当前分支的 `datasets/LSDIR_precrop512/train.iqa_caption_suggestion.jsonl` 已经满足配对训练需要，因此不需要新增人工标注。自动 reward_v1 需要可用的 PyIQA；没有 PyIQA 时可使用 `--no_quality` 做联调，但这不能作为正式 RL 训练奖励。

运行时的文件路径可以因服务器而不同。状态、rollout 和 adapter 的哈希只包含数据内容、根目录下的相对 key、采样器配置和模型张量，不包含绝对路径、时间、GPU 或输出目录。

## A：固定策略与配置

从现有 F0 checkpoint 准备 `rg_flux_adapters` 目录。新配置为 [rl_sr_refinement_flux2_klein_moe.yaml](../configs/rl_sr_refinement_flux2_klein_moe.yaml)，已按 checkpoint-00036000 固化 `iqa_suggestion + caption`、condition8 MoE、rank 32、4 experts、`uniform` flow-matching sigma。只需把其中 `model.flux_model_path` 改成当前服务器的模型位置；这不会影响任何 artifact hash。命令中的 `<F0_ADAPTER>` 可传 checkpoint 根目录或其下 `rg_flux_adapters`；策略哈希会自动只计算后者，不会包含 optimizer/training state。

不要把 `training.freeze_flux_transformer` 改为 `false`。C 和 E 的入口会在创建 optimizer 前检查此项，并将未列入 LoRA/condition adapter 白名单的参数全部冻结。

## B：生成第一轮状态及后续多轮状态

先仅生成 F0 的第 1 轮输出 `y_1`。`--jsonl_path` 必须是和输入 LR 对应的配对 JSONL。

```powershell
python tools/run_rg_flux_iterative_inference.py `
  --checkpoint <F0_ADAPTER> `
  --config configs/rl_sr_refinement_flux2_klein_moe.yaml `
  --input <LQ_ROOT> --jsonl_path <PAIRED_JSONL> `
  --output_dir <RUN_ROOT>/f0_round1 --iterations 1 --upscale 1
```

把 lineage 转换为 C 所需状态。三个 root 只生成相对 key；它们可以按服务器重设。

```powershell
python tools/build_sr_refinement_states.py `
  --lineage_jsonl <RUN_ROOT>/f0_round1/sample_lineage.jsonl `
  --source_jsonl <PAIRED_JSONL> --dataset_id lsdir_precrop512_v1 `
  --lr_root <DATASET_ROOT> --hr_root <DATASET_ROOT> --artifact_root <RUN_ROOT>/f0_round1 `
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
  --input <LQ_ROOT> --jsonl_path <PAIRED_JSONL> `
  --output_dir <RUN_ROOT>/sft_rounds --iterations 4 --upscale 1

python tools/build_sr_refinement_states.py `
  --lineage_jsonl <RUN_ROOT>/sft_rounds/sample_lineage.jsonl `
  --source_jsonl <PAIRED_JSONL> --dataset_id lsdir_precrop512_v1 `
  --lr_root <DATASET_ROOT> --hr_root <DATASET_ROOT> --artifact_root <RUN_ROOT>/sft_rounds `
  --producer_adapter <RUN_ROOT>/g_sft/rg_flux_adapters `
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
