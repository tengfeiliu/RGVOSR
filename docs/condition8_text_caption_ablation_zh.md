# Condition8-as-Text + Caption 消融实验

## 实验定义

四组实验都保留 Caption，并使用相同的 LR 图像条件。训练数据由配置严格读取已经裁剪好的 512×512 RGB 图像；推理命令使用 `--full_frame_inference`，对原始全图推理，不执行训练裁剪。

| 组别 | FLUX 文本输入 | LoRA | Router输入 |
|---|---|---|---|
| A Prompt-Single | Caption + 原始 IQA/Suggestion | Single | 无 |
| B C8Text-Single | Caption + Condition8规范文本 | Single | 无 |
| C Prompt-MoE | Caption + 原始 IQA/Suggestion | MoE | Prompt only |
| D C8Text-MoE | Caption + Condition8规范文本 | MoE | 数值Condition8 |

`condition8_text` 使用固定规则把 `text8_v1` 的八维数值量化为 `no visible/subtle/mild/moderate/severe/extreme`，再送入原有 Text Encoder。它不会把原始 IQA/Suggestion 文本附加到 Prompt。

为了控制初始化，C 从 A 的 checkpoint-24000 初始化；D 必须从 B 的 checkpoint-24000 初始化。

## 公共服务器路径

以下命令假定当前目录是仓库根目录，并且已安装 Conda 环境 `sr-flux2`：

```bash
PROMPT_SINGLE_RUN=exp_rg_flux_sr/rg_flux2_klein_precrop_curriculum_20260729_134536
C8_SINGLE_RUN=exp_rg_flux_sr/ablation_condition8_text_single_seed42_v1

INFERENCE_JSONL=datasets/inference.iqa_caption_suggestion.jsonl
REALLR200=/root/autodl-tmp/datasets/omgsr_eval/RealLR200-20260418T151906Z-3-001/RealLR200
REALLQ250=/root/autodl-tmp/datasets/omgsr_eval/RealLQ250/lq
```

下面每条命令均可独立后台运行。日志采用绝对路径，避免终端关闭后丢失。

## A. Prompt + Single-LoRA（评估已有训练结果）

```bash
nohup env HF_ENDPOINT=https://hf-mirror.com TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
  conda run -n sr-flux2 --no-capture-output \
  python tools/run_rg_flux_pipeline.py \
    --skip_train \
    --run_dir exp_rg_flux_sr/rg_flux2_klein_precrop_curriculum_20260729_134536 \
    --train_config configs/train_rg_flux2_klein_sr_stage0b_512_prompt_curriculum_precropped.yaml \
    --checkpoint_steps 12000 16000 20000 24000 \
    --dataset_dirs \
      RealLR200=/root/autodl-tmp/datasets/omgsr_eval/RealLR200-20260418T151906Z-3-001/RealLR200 \
      RealLQ250=/root/autodl-tmp/datasets/omgsr_eval/RealLQ250/lq \
    --jsonl_path datasets/inference.iqa_caption_suggestion.jsonl \
    --text_encoding_mode online \
    --prompt_variant iqa_suggestion \
    --use_prompt --use_suggestions --include_caption \
    --no-use_degradation_vector \
    --full_frame_inference --restore_input_size \
    --num_inference_steps 25 --upscale 4 --dtype bf16 --device cuda \
    --metrics clipiqa clipiqa+ nima niqe liqe musiq maniqa-pipal \
    --metric_device cuda \
    --run_bad_cases \
    --bad_case_metrics clipiqa maniqa-pipal musiq \
    --bad_case_mode separate --bad_case_worst_k 50 --bad_case_font_size 40 \
  > /root/autodl-tmp/ablation_prompt_single.log 2>&1 < /dev/null &
```

## B. Condition8-as-Text + Single-LoRA

```bash
nohup env HF_ENDPOINT=https://hf-mirror.com TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
  conda run -n sr-flux2 --no-capture-output \
  python tools/run_rg_flux_pipeline.py \
    --train_config configs/train_rg_flux2_klein_sr_stage0b_512_condition8_text_caption_precropped.yaml \
    --exp_name ablation_condition8_text_single_seed42_v1 \
    --accelerate_config configs/accelerate/zero3_bf16_param_offload.yaml \
    --num_processes 1 --max_steps 24000 \
    --checkpoint_steps 12000 16000 20000 24000 \
    --dataset_dirs \
      RealLR200=/root/autodl-tmp/datasets/omgsr_eval/RealLR200-20260418T151906Z-3-001/RealLR200 \
      RealLQ250=/root/autodl-tmp/datasets/omgsr_eval/RealLQ250/lq \
    --jsonl_path datasets/inference.iqa_caption_suggestion.jsonl \
    --text_encoding_mode online \
    --prompt_variant condition8_text \
    --use_prompt --use_suggestions --include_caption \
    --no-use_degradation_vector \
    --full_frame_inference --restore_input_size \
    --num_inference_steps 25 --upscale 4 --dtype bf16 --device cuda \
    --metrics clipiqa clipiqa+ nima niqe liqe musiq maniqa-pipal \
    --metric_device cuda \
    --run_bad_cases \
    --bad_case_metrics clipiqa maniqa-pipal musiq \
    --bad_case_mode separate --bad_case_worst_k 50 --bad_case_font_size 40 \
  > /root/autodl-tmp/ablation_condition8_text_single.log 2>&1 < /dev/null &
```

## C. Prompt + MoE-LoRA

```bash
nohup env HF_ENDPOINT=https://hf-mirror.com TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
  conda run -n sr-flux2 --no-capture-output \
  python tools/run_rg_flux_moe_pipeline.py \
    --moe_config configs/train_rg_flux2_klein_sr_moe_stage0b_512_prompt_curriculum_precropped.yaml \
    --exp_name ablation_prompt_moe_seed42_v1 \
    --single_lora_run_dir exp_rg_flux_sr/rg_flux2_klein_precrop_curriculum_20260729_134536 \
    --single_lora_checkpoint_step 24000 \
    --accelerate_config configs/accelerate/zero3_bf16_param_offload.yaml \
    --num_processes 1 \
    --prototype_num_samples 128 --perturb_scale 0.3 --init_device cuda \
    --router_input_mode prompt_only \
    --checkpoint_steps 12000 16000 20000 24000 30000 36000 40000 \
    --dataset_dirs \
      RealLR200=/root/autodl-tmp/datasets/omgsr_eval/RealLR200-20260418T151906Z-3-001/RealLR200 \
      RealLQ250=/root/autodl-tmp/datasets/omgsr_eval/RealLQ250/lq \
    --jsonl_path datasets/inference.iqa_caption_suggestion.jsonl \
    --text_encoding_mode online \
    --prompt_variant iqa_suggestion \
    --use_prompt --use_suggestions --include_caption \
    --no-use_degradation_vector \
    --full_frame_inference --restore_input_size \
    --num_inference_steps 25 --upscale 4 --dtype bf16 --device cuda \
    --metrics clipiqa clipiqa+ nima niqe liqe musiq maniqa-pipal \
    --metric_device cuda \
    --run_bad_cases \
    --bad_case_metrics clipiqa maniqa-pipal musiq \
    --bad_case_mode separate --bad_case_worst_k 50 --bad_case_font_size 40 \
  > /root/autodl-tmp/ablation_prompt_moe.log 2>&1 < /dev/null &
```

## D. Condition8-as-Text + MoE-LoRA

必须等待 B 至少生成 `checkpoint-00024000` 后再运行。

```bash
nohup env HF_ENDPOINT=https://hf-mirror.com TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
  conda run -n sr-flux2 --no-capture-output \
  python tools/run_rg_flux_moe_pipeline.py \
    --moe_config configs/train_rg_flux2_klein_sr_moe_stage0b_512_condition8_text_caption_precropped.yaml \
    --exp_name ablation_condition8_text_moe_seed42_v1 \
    --single_lora_run_dir exp_rg_flux_sr/ablation_condition8_text_single_seed42_v1 \
    --single_lora_checkpoint_step 24000 \
    --accelerate_config configs/accelerate/zero3_bf16_param_offload.yaml \
    --num_processes 1 \
    --prototype_num_samples 128 --perturb_scale 0.3 --init_device cuda \
    --router_input_mode condition8 \
    --checkpoint_steps 12000 16000 20000 24000 30000 36000 40000 \
    --dataset_dirs \
      RealLR200=/root/autodl-tmp/datasets/omgsr_eval/RealLR200-20260418T151906Z-3-001/RealLR200 \
      RealLQ250=/root/autodl-tmp/datasets/omgsr_eval/RealLQ250/lq \
    --jsonl_path datasets/inference.iqa_caption_suggestion.jsonl \
    --text_encoding_mode online \
    --prompt_variant condition8_text \
    --use_prompt --use_suggestions --include_caption \
    --no-use_degradation_vector \
    --full_frame_inference --restore_input_size \
    --num_inference_steps 25 --upscale 4 --dtype bf16 --device cuda \
    --metrics clipiqa clipiqa+ nima niqe liqe musiq maniqa-pipal \
    --metric_device cuda \
    --run_bad_cases \
    --bad_case_metrics clipiqa maniqa-pipal musiq \
    --bad_case_mode separate --bad_case_worst_k 50 --bad_case_font_size 40 \
  > /root/autodl-tmp/ablation_condition8_text_moe.log 2>&1 < /dev/null &
```

## 查看后台任务

```bash
tail -f /root/autodl-tmp/ablation_condition8_text_single.log
tail -f /root/autodl-tmp/ablation_prompt_moe.log
tail -f /root/autodl-tmp/ablation_condition8_text_moe.log
```

## 推理与评估同步说明

- `inference_rg_flux_sr.py` 支持 `--prompt_variant condition8_text`，并从全图 JSONL 获取 Caption、IQA 和 Suggestion。
- Condition8规范文本与训练阶段使用同一个 `build_sr_prompt()` 实现，不存在训练/推理模板漂移。
- 推理 manifest 记录每张图的精确 Prompt，并额外记录 `prompt_variant` 和 `include_caption`。
- `eval_rg_flux_sr_metrics.py` 只评估推理输出图像，不依赖 Prompt，因此指标实现无需修改。
- bad-case 优先读取推理时保存的精确 Prompt；旧 manifest 回退重建时也会保留 Caption 和 `condition8_text`。
- 新 Prompt hash 与旧缓存不同，第一轮实验建议使用 `--text_encoding_mode online`。如需 cached 模式，应为四组分别重建缓存。
