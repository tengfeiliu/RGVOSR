# E0/E2 与推理采样消融命令

## 实验定义

- E0：`sigma_sampling=uniform`。
- E2：`sigma_sampling=logit_normal`，其中 `mean=-0.4`、`std=1.0`。
- 四组训练均使用 `condition8_text + caption`、`batch_size=1`、`grad_accum_steps=4`、`lpips_weight=0.1`。
- 训练后的默认推理基线固定为 25 步、linear schedule、pure noise、`sigma_start=1.0`。

以下命令假定从仓库根目录执行，并使用同一组验证数据：

```bash
REALLR200=/root/autodl-tmp/datasets/omgsr_eval/RealLR200-20260418T151906Z-3-001/RealLR200
REALLQ250=/root/autodl-tmp/datasets/omgsr_eval/RealLQ250/lq
INFERENCE_JSONL=datasets/inference.iqa_caption_suggestion.jsonl
ACCELERATE_CONFIG=configs/accelerate/zero3_bf16_param_offload.yaml
```

## 1. Single E0

```bash
python tools/run_rg_flux_pipeline.py \
  --train_config configs/train_rg_flux2_klein_sr_stage0b_512_condition8_text_caption_precropped.yaml \
  --exp_name rgflux_c8_single_e0_uniform_seed42 \
  --accelerate_config "$ACCELERATE_CONFIG" --num_processes 1 \
  --checkpoint_steps 4000 8000 12000 16000 20000 24000 28000 \
  --dataset_dirs RealLR200="$REALLR200" RealLQ250="$REALLQ250" \
  --jsonl_path "$INFERENCE_JSONL" --text_encoding_mode online \
  --prompt_variant condition8_text --use_prompt --use_suggestions --include_caption \
  --no-use_degradation_vector --full_frame_inference --restore_input_size \
  --num_inference_steps 25 --inference_schedule linear \
  --inference_init_mode pure_noise --inference_sigma_start 1.0 \
  --upscale 4 --dtype bf16 --device cuda \
  --metrics clipiqa clipiqa+ nima niqe liqe musiq maniqa-pipal --metric_device cuda
```

## 2. Single E2

```bash
python tools/run_rg_flux_pipeline.py \
  --train_config configs/train_rg_flux2_klein_sr_stage0b_512_condition8_text_caption_precropped_e2_lognorm_m04.yaml \
  --exp_name rgflux_c8_single_e2_lognorm_m04_seed42 \
  --accelerate_config "$ACCELERATE_CONFIG" --num_processes 1 \
  --checkpoint_steps 4000 8000 12000 16000 20000 24000 28000 \
  --dataset_dirs RealLR200="$REALLR200" RealLQ250="$REALLQ250" \
  --jsonl_path "$INFERENCE_JSONL" --text_encoding_mode online \
  --prompt_variant condition8_text --use_prompt --use_suggestions --include_caption \
  --no-use_degradation_vector --full_frame_inference --restore_input_size \
  --num_inference_steps 25 --inference_schedule linear \
  --inference_init_mode pure_noise --inference_sigma_start 1.0 \
  --upscale 4 --dtype bf16 --device cuda \
  --metrics clipiqa clipiqa+ nima niqe liqe musiq maniqa-pipal --metric_device cuda
```

## 3. MoE E0

此命令使用对应的 Single E0 最终 checkpoint 初始化 MoE。

```bash
python tools/run_rg_flux_moe_pipeline.py \
  --moe_config configs/train_rg_flux2_klein_sr_moe_stage0b_512_condition8_text_caption_precropped.yaml \
  --exp_name rgflux_c8_moe_e0_uniform_seed42 \
  --single_lora_run_dir exp_rg_flux_sr/rgflux_c8_single_e0_uniform_seed42 \
  --single_lora_checkpoint_step 28000 \
  --accelerate_config "$ACCELERATE_CONFIG" --num_processes 1 \
  --prototype_num_samples 128 --perturb_scale 0.3 --init_device cuda \
  --router_input_mode condition8 \
  --checkpoint_steps 4000 8000 12000 16000 20000 24000 28000 32000 36000 40000 \
  --dataset_dirs RealLR200="$REALLR200" RealLQ250="$REALLQ250" \
  --jsonl_path "$INFERENCE_JSONL" --text_encoding_mode online \
  --prompt_variant condition8_text --use_prompt --use_suggestions --include_caption \
  --no-use_degradation_vector --full_frame_inference --restore_input_size \
  --num_inference_steps 25 --inference_schedule linear \
  --inference_init_mode pure_noise --inference_sigma_start 1.0 \
  --upscale 4 --dtype bf16 --device cuda \
  --metrics clipiqa clipiqa+ nima niqe liqe musiq maniqa-pipal --metric_device cuda
```

## 4. MoE E2

此命令使用对应的 Single E2 最终 checkpoint 初始化 MoE，因此衡量的是 E2 在完整 Single→MoE 链路上的累计效果。

```bash
python tools/run_rg_flux_moe_pipeline.py \
  --moe_config configs/train_rg_flux2_klein_sr_moe_stage0b_512_condition8_text_caption_precropped_e2_lognorm_m04.yaml \
  --exp_name rgflux_c8_moe_e2_lognorm_m04_seed42 \
  --single_lora_run_dir exp_rg_flux_sr/rgflux_c8_single_e2_lognorm_m04_seed42 \
  --single_lora_checkpoint_step 28000 \
  --accelerate_config "$ACCELERATE_CONFIG" --num_processes 1 \
  --prototype_num_samples 128 --perturb_scale 0.3 --init_device cuda \
  --router_input_mode condition8 \
  --checkpoint_steps 4000 8000 12000 16000 20000 24000 28000 32000 36000 40000 \
  --dataset_dirs RealLR200="$REALLR200" RealLQ250="$REALLQ250" \
  --jsonl_path "$INFERENCE_JSONL" --text_encoding_mode online \
  --prompt_variant condition8_text --use_prompt --use_suggestions --include_caption \
  --no-use_degradation_vector --full_frame_inference --restore_input_size \
  --num_inference_steps 25 --inference_schedule linear \
  --inference_init_mode pure_noise --inference_sigma_start 1.0 \
  --upscale 4 --dtype bf16 --device cuda \
  --metrics clipiqa clipiqa+ nima niqe liqe musiq maniqa-pipal --metric_device cuda
```

若要只隔离“MoE 阶段 sigma 采样”的影响，应让 MoE E0/E2 都从同一个 Single checkpoint 初始化；将两条 MoE 命令的 `--single_lora_run_dir` 和 `--single_lora_checkpoint_step` 设成相同值即可。

## 5. 推理 2×2 对比：schedule × initialization

先选定同一个 run、checkpoint、seed 和数据集。不要跨训练 checkpoint 比较推理 sampler。

```bash
RUN_DIR=exp_rg_flux_sr/rgflux_c8_single_e2_lognorm_m04_seed42
TRAIN_CONFIG=configs/train_rg_flux2_klein_sr_stage0b_512_condition8_text_caption_precropped_e2_lognorm_m04.yaml
CKPT=28000

COMMON_INFER_ARGS=(
  --run_dir "$RUN_DIR" --checkpoint_step "$CKPT" --config "$TRAIN_CONFIG"
  --dataset_dirs RealLR200="$REALLR200" RealLQ250="$REALLQ250"
  --jsonl_path "$INFERENCE_JSONL" --text_encoding_mode online
  --prompt_variant condition8_text --use_prompt --use_suggestions --include_caption
  --no-use_degradation_vector --full_frame_inference --restore_input_size
  --num_inference_steps 25 --upscale 4 --dtype bf16 --device cuda --seed 42
)
```

Linear + pure noise（基线）：

```bash
python inference_rg_flux_sr.py "${COMMON_INFER_ARGS[@]}" \
  --output_dir "$RUN_DIR/inference_ablation/linear_pure_noise" \
  --inference_schedule linear --inference_init_mode pure_noise --inference_sigma_start 1.0
```

Empirical shift + pure noise（与上一条隔离 schedule）：

```bash
python inference_rg_flux_sr.py "${COMMON_INFER_ARGS[@]}" \
  --output_dir "$RUN_DIR/inference_ablation/empirical_shift_pure_noise" \
  --inference_schedule empirical_shift --inference_init_mode pure_noise --inference_sigma_start 1.0
```

Linear + LR warm-start：

```bash
python inference_rg_flux_sr.py "${COMMON_INFER_ARGS[@]}" \
  --output_dir "$RUN_DIR/inference_ablation/linear_lr_warm_start_s08" \
  --inference_schedule linear --inference_init_mode lr_warm_start --inference_sigma_start 0.8
```

Empirical shift + LR warm-start：

```bash
python inference_rg_flux_sr.py "${COMMON_INFER_ARGS[@]}" \
  --output_dir "$RUN_DIR/inference_ablation/empirical_shift_lr_warm_start_s08" \
  --inference_schedule empirical_shift --inference_init_mode lr_warm_start --inference_sigma_start 0.8
```

`sigma_start=0.8` 是首轮实验点，不应直接视作最优值。如果 warm-start 胜出，再固定 checkpoint、schedule 和 seed，小范围测试 `0.7/0.8/0.9`。每个 `inference_manifest.json` 都会记录 schedule、init mode、sigma start、步数和 seed。
