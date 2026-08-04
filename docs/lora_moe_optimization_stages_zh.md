# LoRA-MoE 分阶段优化、训练与全图推理说明

## 1. 本轮改动的边界

目标方案 S2–S5 只从 Router 中移除 LR 图像编码；S1 保留旧输入，作为修复后的
对照基线。FLUX.2-klein 的 SR 主干始终通过
`flux2_image_concat` 接收 `z_lr`，因此不会丢失图像恢复条件。

旧的 `degradation_vector` 已确认无效，本方案不会读取或回退到该字段。
新的 Router 条件由训练与推理共用的 `text8_v1` 抽取器，从最终生效的
`profile.iqa` 和 `profile.suggestion` 确定性生成。

8 维依次为：

1. blur：模糊；
2. noise：噪声；
3. compression：压缩块/JPEG；
4. ringing_aliasing：振铃、光晕、锯齿、摩尔纹；
5. texture_loss：纹理/细节损失；
6. photometric：曝光、颜色、对比度问题；
7. structure_risk：文字、人脸、细线、几何等结构保真风险；
8. hallucination_risk：伪细节、过锐化、语义改变风险。

抽取器同时产生 `valid_mask`、覆盖率相关的 `confidence`、版本号和源文本哈希。
缺失与明确的“无该退化”不会混为一谈；全缺失样本不接受 Teacher 强制路由，
而是退回学习 Router。固定的 Suggestion 保真尾句会被剥离，避免所有样本得到
同一个伪结构标签。

## 2. 训练与推理的空间处理

- 训练配置固定为 `data.pre_cropped: true`、`crop_size: 512`。HQ、LQ 必须都已经是
  RGB 512×512，Dataset 不会再次随机裁剪；对应 Prompt 来自训练 JSONL。
- 独立推理必须传 `--full_frame_inference`。它只覆盖推理期的 `pre_cropped`，不会
  改写训练配置，也不会把原图中心裁成 512×512。
- 当前全图推理会先把原 LQ 按 `upscale=4` 双三次放大，再对齐到 VAE 所需的 16 倍数；
  对齐实现可能轻微缩小边缘尺寸。`--restore_input_size` 最终把输出恢复为
  `原始 LQ 宽×4 × 原始 LQ 高×4`。这是“全图、不裁剪”，但不是完全不缩放预处理。
- 全图推理 JSONL 必须提供与原图匹配的 IQA/Suggestion。IQA 或 Suggestion shuffle 时，
  8 维条件从 shuffle 后真正用于 Prompt 的 `paired_profile` 提取，避免文本与 Router 不一致。

## 3. 建议的独立消融阶段

每个阶段都从同一个 Single-LoRA `checkpoint-24000` 独立初始化，不能串联前一阶段的
MoE checkpoint。Single-LoRA 与 MoE 使用相同训练集是正确且更公平的对照。

| 阶段 | Router 输入 | Teacher | 目的 |
|---|---|---:|---|
| S0 | Single-LoRA model-only continued | 不适用 | 同样 fresh optimizer，对齐额外 40k steps 的容量基线 |
| S1 | Prompt + LR latent 统计/卷积 | 否 | 只验证初始化、EMA balance、探索和 warmup 修复 |
| S2 | Prompt | 否 | 去掉 Router 的 LR 编码，判断旧 LR 分支是否有贡献 |
| S3 | 结构化 8 维 IQA/Suggestion | 否 | KMeans prototype；只判断高信噪比条件是否优于 Prompt mean-pool |
| S4 | 结构化 8 维 | 是 | 保持 KMeans prototype；前 15% 纯 Teacher，15%–35% 线性移交 Router |
| S5 | 结构化 8 维 + raw sigma | 是 | 保持 KMeans prototype；增加结构/纹理恢复阶段分化 |

所有阶段共同使用：`perturb_scale=0.3`、30% soft warmup、早期 noisy exploration、
跨 step EMA balance、Top-2 部署路由、有效增量 `BA` 的功能多样性，以及
`grad_accum_steps=8`。EMA balance 当前只保证单进程正确，因此命令固定
`--num_processes 1`。

### S0：Single-LoRA model-only + fresh-optimizer 对照

原始 Single-LoRA checkpoint 是 step 24000，而 MoE 会从该权重额外训练 40000 个
optimizer steps。不能只拿原始 step 24000 与 MoE step 40000 比。下面命令从 Single
checkpoint-24000 恢复模型权重和 global step，但像 MoE Stage1 一样使用 fresh optimizer/
scheduler，再训练到总 step 64000；因此 Single 的
36000/40000/…/64000 分别对应 MoE 的额外 12000/16000/…/40000 steps。
两边均使用 batch=1、grad accumulation=8 和相同数据。这里刻意不恢复原 Single-LoRA
optimizer，避免拿“已训练 24k 的 optimizer 状态”对比 MoE 的 fresh optimizer。运行前仍须
存在 `training_state.pt` 或 `training_state_rank-00000.pt`，用于读取 global step。
S0 还显式把 `loss.image_loss_crop_size` 对齐为 MoE 的 256；训练输入本身仍是完整的
预裁剪 512×512。

```bash
cd /root/autodl-tmp/RGVOSR   # 按服务器实际仓库路径修改
mkdir -p logs/moe_stages pids

SINGLE_CKPT=exp_rg_flux_sr/rg_flux2_klein_precrop_curriculum_20260729_134536/checkpoints/checkpoint-00024000
test -f "$SINGLE_CKPT/training_state.pt" -o -f "$SINGLE_CKPT/training_state_rank-00000.pt" || {
  echo "缺少Single-LoRA global-step训练状态，不能运行continued基线"; return 1 2>/dev/null || exit 1;
}

S0_LOG="logs/moe_stages/s0_single_continue-$(date +%Y%m%d-%H%M%S).log"
nohup env \
  HF_ENDPOINT=https://hf-mirror.com \
  TOKENIZERS_PARALLELISM=false \
  PYTHONUNBUFFERED=1 \
  conda run -n sr-flux2 --no-capture-output \
    python tools/run_rg_flux_pipeline.py \
      --train_config configs/train_rg_flux2_klein_sr_stage0b_512_prompt_curriculum_precropped.yaml \
      --resume_checkpoint "$SINGLE_CKPT" \
      --no-resume_training_state \
      --max_steps 64000 \
      --grad_accum_steps 8 \
      --image_loss_crop_size 256 \
      --stage_label _s0_single_continue_control \
      --accelerate_config configs/accelerate/zero3_bf16_param_offload.yaml \
      --num_processes 1 \
      --checkpoint_steps 36000 40000 44000 48000 54000 60000 64000 \
      --dataset_dirs \
        RealLR200=/root/autodl-tmp/datasets/omgsr_eval/RealLR200-20260418T151906Z-3-001/RealLR200 \
        RealLQ250=/root/autodl-tmp/datasets/omgsr_eval/RealLQ250/lq \
      --jsonl_path datasets/inference.iqa_caption_suggestion.jsonl \
      --text_encoding_mode online \
      --prompt_variant iqa_suggestion \
      --use_prompt --use_suggestions --no-use_degradation_vector \
      --full_frame_inference --restore_input_size --upscale 4 \
      --num_inference_steps 25 --dtype bf16 --device cuda \
      --metrics clipiqa clipiqa+ nima niqe liqe musiq maniqa-pipal \
      --metric_device cuda --run_bad_cases \
      --bad_case_metrics clipiqa maniqa-pipal musiq \
      --bad_case_mode separate --bad_case_worst_k 50 --bad_case_font_size 40 \
    > "$S0_LOG" 2>&1 < /dev/null &

echo $! > pids/s0_single_continue.pid
echo "S0 PID=$(cat pids/s0_single_continue.pid), log=$S0_LOG"
```

## 4. 服务器后台一键命令

先进入仓库根目录，然后定义公共参数。以下路径已按当前服务器信息填写；如果基础模型
不在配置中的 `/root/autodl-tmp/models/FLUX.2-klein-base-4B/`，需先修改 Single 与
MoE 两份 YAML 的 `model.flux_model_path`，并确保二者指向同一份基础权重。

```bash
cd /root/autodl-tmp/RGVOSR   # 按服务器实际仓库路径修改

mkdir -p logs/moe_stages pids

COMMON_ARGS=(
  --moe_config configs/train_rg_flux2_klein_sr_moe_stage0b_512_prompt_curriculum_precropped.yaml
  --single_lora_run_dir exp_rg_flux_sr/rg_flux2_klein_precrop_curriculum_20260729_134536
  --single_lora_checkpoint_step 24000
  --accelerate_config configs/accelerate/zero3_bf16_param_offload.yaml
  --num_processes 1
  --prototype_num_samples 128
  --perturb_scale 0.3
  --init_device cuda
  --init_seed 42
  --expert_init_seed 42
  --checkpoint_steps 12000 16000 20000 24000 30000 36000 40000
  --dataset_dirs
    RealLR200=/root/autodl-tmp/datasets/omgsr_eval/RealLR200-20260418T151906Z-3-001/RealLR200
    RealLQ250=/root/autodl-tmp/datasets/omgsr_eval/RealLQ250/lq
  --jsonl_path datasets/inference.iqa_caption_suggestion.jsonl
  --text_encoding_mode online
  --prompt_variant iqa_suggestion
  --use_prompt
  --use_suggestions
  --no-use_degradation_vector
  --full_frame_inference
  --restore_input_size
  --num_inference_steps 25
  --upscale 4
  --dtype bf16
  --device cuda
  --metrics clipiqa clipiqa+ nima niqe liqe musiq maniqa-pipal
  --metric_device cuda
  --run_bad_cases
  --bad_case_metrics clipiqa maniqa-pipal musiq
  --bad_case_mode separate
  --bad_case_worst_k 50
  --bad_case_font_size 40
)

launch_stage () {
  local label="$1"
  local router_mode="$2"
  local teacher_flag="$3"
  local teacher_weight="$4"
  local semantic_prototype_flag="$5"
  local log_file="logs/moe_stages/${label}-$(date +%Y%m%d-%H%M%S).log"

  nohup env \
    HF_ENDPOINT=https://hf-mirror.com \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    conda run -n sr-flux2 --no-capture-output \
      python tools/run_rg_flux_moe_pipeline.py \
      "${COMMON_ARGS[@]}" \
      --router_input_mode "$router_mode" \
      "$teacher_flag" \
      --router_teacher_weight "$teacher_weight" \
      "$semantic_prototype_flag" \
      --stage_label "_${label}" \
      > "$log_file" 2>&1 < /dev/null &

  local pid=$!
  echo "$pid" > "pids/${label}.pid"
  echo "${label}: PID=${pid}, log=${log_file}"
}
```

依次运行下面五条命令。单卡上不要同时启动多个阶段；一个阶段完成后再启动下一个。

```bash
# S1：稳定性修复基线，保留旧 Prompt+LR Router
launch_stage s1_stable_prompt_lr prompt_lr --no-teacher_routing_enabled 0.0 --no-semantic_prototype_init

# S2：只去掉 Router 的 LR 图像编码
launch_stage s2_prompt_only prompt_only --no-teacher_routing_enabled 0.0 --no-semantic_prototype_init

# S3：8维结构化条件，不使用 Teacher
launch_stage s3_condition8 condition8 --no-teacher_routing_enabled 0.0 --no-semantic_prototype_init

# S4：8维结构化条件 + Teacher curriculum
launch_stage s4_condition8_teacher condition8 --teacher_routing_enabled 0.01 --no-semantic_prototype_init

# S5：8维结构化条件 + Teacher + timestep
launch_stage s5_condition8_teacher_timestep condition8_timestep --teacher_routing_enabled 0.01 --no-semantic_prototype_init
```

每条命令会自动完成：Single-LoRA→MoE 初始化、512×512 预裁剪训练、多 checkpoint
原图推理、指标评估和 bad-case。查看状态：

```bash
tail -f logs/moe_stages/s4_condition8_teacher-*.log
ps -fp "$(cat pids/s4_condition8_teacher.pid)"
```

正式训练前，在服务器的 `sr-flux2` 环境运行 Router 与 CLI 回归测试。以下测试不应显示
`skipped: torch is not installed`：

```bash
TOKENIZERS_PARALLELISM=false conda run -n sr-flux2 --no-capture-output \
  python -m unittest \
    tests.test_lora_moe \
    tests.test_router_condition \
    tests.test_rg_flux_moe_pipeline_cli \
    tests.test_rg_flux_pipeline_cli -v
```

## 5. 中断后恢复训练并继续全图评估

先指定要恢复的 MoE run 目录。运行时配置由 pipeline 保存在 run 根目录。

```bash
MOE_RUN_DIR=/root/autodl-tmp/RGVOSR/exp_rg_flux_sr/你的MoE运行目录
RESUME_CONFIG="$MOE_RUN_DIR/resume_runtime_config.yaml"
cp "$MOE_RUN_DIR/pipeline_runtime_config.yaml" "$RESUME_CONFIG"

# 从该run最新checkpoint恢复模型、optimizer、scheduler和Router EMA，而不是重新加载Stage1。
sed -i -E \
  -e 's#^  resume_ckpt:.*#  resume_ckpt: null#' \
  -e 's#^  auto_resume:.*#  auto_resume: true#' \
  -e 's#^  resume_training_state:.*#  resume_training_state: true#' \
  "$RESUME_CONFIG"

mkdir -p "$MOE_RUN_DIR/logs"
RESUME_LOG="$MOE_RUN_DIR/logs/resume-$(date +%Y%m%d-%H%M%S).log"

nohup env \
  HF_ENDPOINT=https://hf-mirror.com \
  TOKENIZERS_PARALLELISM=false \
  PYTHONUNBUFFERED=1 \
  conda run -n sr-flux2 --no-capture-output \
    accelerate launch \
      --config_file configs/accelerate/zero3_bf16_param_offload.yaml \
      --num_processes 1 \
      train_rg_flux_sr.py --config "$RESUME_CONFIG" \
    > "$RESUME_LOG" 2>&1 < /dev/null &

echo $! > "$MOE_RUN_DIR/resume.pid"
echo "log=$RESUME_LOG"
```

恢复训练完成后，对已有 checkpoints 单独执行全图推理、评估和 bad-case：

```bash
POST_LOG="$MOE_RUN_DIR/logs/post-eval-$(date +%Y%m%d-%H%M%S).log"

nohup env \
  HF_ENDPOINT=https://hf-mirror.com \
  TOKENIZERS_PARALLELISM=false \
  PYTHONUNBUFFERED=1 \
  conda run -n sr-flux2 --no-capture-output \
    python tools/run_rg_flux_moe_pipeline.py \
      --skip_stage1 \
      --skip_train \
      --moe_run_dir "$MOE_RUN_DIR" \
      --moe_config "$RESUME_CONFIG" \
      --checkpoint_steps 12000 16000 20000 24000 30000 36000 40000 \
      --dataset_dirs \
        RealLR200=/root/autodl-tmp/datasets/omgsr_eval/RealLR200-20260418T151906Z-3-001/RealLR200 \
        RealLQ250=/root/autodl-tmp/datasets/omgsr_eval/RealLQ250/lq \
      --jsonl_path datasets/inference.iqa_caption_suggestion.jsonl \
      --text_encoding_mode online \
      --prompt_variant iqa_suggestion \
      --use_prompt \
      --use_suggestions \
      --no-use_degradation_vector \
      --full_frame_inference \
      --restore_input_size \
      --num_inference_steps 25 \
      --upscale 4 \
      --dtype bf16 \
      --device cuda \
      --metrics clipiqa clipiqa+ nima niqe liqe musiq maniqa-pipal \
      --metric_device cuda \
      --run_bad_cases \
      --bad_case_metrics clipiqa maniqa-pipal musiq \
      --bad_case_mode separate \
      --bad_case_worst_k 50 \
      --bad_case_font_size 40 \
    > "$POST_LOG" 2>&1 < /dev/null &

echo $! > "$MOE_RUN_DIR/post-eval.pid"
echo "log=$POST_LOG"
```

## 6. 对比与停止条件

不要只看 NR-IQA 均值。至少同时检查：各数据集指标、bad-case、Router entropy、
`expert_i_usage`、`expert_i_used`、`expert_i_ema`，以及同一图的主观伪细节/结构保真。
若 S3 不优于 S2，说明文本 8 维抽取覆盖率或专家语义矩阵需要调整；若 S4 优于 S3，
Teacher 确实促进了分化；若 S5 只改变路由但不提升指标，应先查看不同 sigma 的专家使用率，
不要直接继续增加模型复杂度。

中间 checkpoint 的独立推理统一使用最终部署路由（Top-2、temperature=0.7），不是该
checkpoint 训练当时的插值温度。具体推理 schedule 会写入
`inference_manifest.json`，因此比较是可追溯的。

本轮没有实现 relation loss。这不是隐藏开关：batch=1 且当前数据加载器没有显式的
“同一 HR、不同退化”成对采样，直接计算 relation loss 没有可靠监督对象。待补充 pair
sampler、pair metadata 和有效 pair 计数后，再把它作为 S6；当前命令不会假装启用它。

正式阶段每个需要 40k steps，成本较高。建议先复制配置做 1k–2k steps smoke run，验证
数据覆盖、显存、checkpoint、全图推理和日志后，再执行本文正式命令；smoke 结果不用于
最终效果结论。
