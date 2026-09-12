#!/usr/bin/env bash
# One-command A–E RL-SR cycle. Data input, inference, and evaluation are all
# derived from config.data.jsonl_path; only the server-local F0 checkpoint is
# required. Usage: bash tools/run_rl_sr_stage_ae.sh [all|f0|c|multiround|reward|e|eval]

set -Eeuo pipefail

STAGE="${1:-all}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -n "${CONDA_ENV:-}" ]]; then
  PYTHON_CMD=(conda run --no-capture-output -n "${CONDA_ENV}" python)
else
  PYTHON_CMD=("${PYTHON_BIN}")
fi
CONFIG="${CONFIG:-${REPO_ROOT}/configs/rl_sr_refinement_flux2_klein_moe.yaml}"

# F0 is intentionally the sole machine-specific required value. Dataset paths
# come from config.data.jsonl_path, and output names are created automatically.
: "${F0_CHECKPOINT:?Set F0_CHECKPOINT to checkpoint-00036000 or its rg_flux_adapters directory}"
F0_ADAPTER="${F0_ADAPTER:-${F0_CHECKPOINT}/rg_flux_adapters}"
if [[ -f "${F0_CHECKPOINT}/flux2_klein_lora_moe_state.pt" ]]; then
  F0_ADAPTER="${F0_CHECKPOINT}"
fi

cd "${REPO_ROOT}"

require_file() {
  if [[ ! -e "$1" ]]; then
    echo "Required artifact does not exist: $1" >&2
    exit 2
  fi
}

config_field() {
  "${PYTHON_CMD[@]}" tools/resolve_rl_sr_run_dir.py \
    --config "${CONFIG}" --f0_checkpoint "${F0_CHECKPOINT}" --repo_root "${REPO_ROOT}" \
    --field "$1"
}

require_file "${CONFIG}"
require_file "${F0_ADAPTER}"

TRAIN_JSONL="$(config_field data_jsonl_path)"
EVAL_JSONL="$(config_field evaluation_jsonl_path)"
ITERATIONS="${ITERATIONS:-$(config_field iterations)}"
NUM_CANDIDATES="${NUM_CANDIDATES:-$(config_field num_candidates)}"
SFT_MAX_STEPS="${SFT_MAX_STEPS:-$(config_field sft_max_steps)}"
RL_MAX_STEPS="${RL_MAX_STEPS:-$(config_field rl_max_steps)}"
SEED="${SEED:-$(config_field seed)}"
ITER_METRIC_DEVICE="${ITER_METRIC_DEVICE:-$(config_field metric_device)}"
REWARD_DEVICE="${REWARD_DEVICE:-$(config_field reward_device)}"
DATASET_ID="${DATASET_ID:-$(config_field dataset_id)}"
read -r -a ITER_METRICS <<< "$(config_field metrics)"

require_file "${TRAIN_JSONL}"
require_file "${EVAL_JSONL}"
if [[ -n "${RL_SR_RUN_DIR:-}" ]]; then
  RUN_ROOT="${RL_SR_RUN_DIR}"
  require_file "${RUN_ROOT}"
else
  RUN_DIR_ARGS=(
    --config "${CONFIG}" --f0_checkpoint "${F0_CHECKPOINT}" --repo_root "${REPO_ROOT}"
  )
  if [[ -n "${RL_SR_OUTPUT_ROOT:-}" ]]; then
    RUN_DIR_ARGS+=(--output_root "${RL_SR_OUTPUT_ROOT}")
  fi
  RUN_ROOT="$("${PYTHON_CMD[@]}" tools/resolve_rl_sr_run_dir.py "${RUN_DIR_ARGS[@]}" --create)"
fi
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
LOG_DIR="${RUN_ROOT}/logs"
INPUT_DIR="${RUN_ROOT}/00_inputs"
mkdir -p "${LOG_DIR}" "${INPUT_DIR}"

TRAIN_INPUT="${INPUT_DIR}/train_lq_inputs.txt"
EVAL_INPUT="${INPUT_DIR}/evaluation_lq_inputs.txt"

echo "RL-SR run directory: ${RUN_ROOT}"
echo "Training data JSONL (config.data.jsonl_path): ${TRAIN_JSONL}"
echo "Evaluation data JSONL: ${EVAL_JSONL}"

run_stage() {
  local name="$1"
  shift
  local log_path="${LOG_DIR}/${name}.log"
  local start_message
  start_message="[$(date '+%F %T')] START ${name}"
  printf '\n%s\n' "${start_message}" >> "${log_path}"
  printf '%s\n' "${start_message}"
  if "$@" >> "${log_path}" 2>&1; then
    local done_message="[$(date '+%F %T')] DONE ${name}"
    printf '%s\n' "${done_message}" >> "${log_path}"
    printf '%s\n' "${done_message}"
  else
    local status=$?
    local failed_message="[$(date '+%F %T')] FAILED ${name}; inspect ${log_path}"
    printf '%s\n' "${failed_message}" >> "${log_path}"
    printf '%s\n' "${failed_message}" >&2
    return "${status}"
  fi
}

prepare_inputs() {
  run_stage "00_create_train_input_manifest" "${PYTHON_CMD[@]}" tools/create_sr_input_manifest.py \
    --data_jsonl_path "${TRAIN_JSONL}" --output "${TRAIN_INPUT}" --label training
  if [[ "${EVAL_JSONL}" == "${TRAIN_JSONL}" ]]; then
    EVAL_INPUT="${TRAIN_INPUT}"
  else
    run_stage "00_create_evaluation_input_manifest" "${PYTHON_CMD[@]}" tools/create_sr_input_manifest.py \
      --data_jsonl_path "${EVAL_JSONL}" --output "${EVAL_INPUT}" --label evaluation
  fi
}

run_f0() {
  run_stage "01_f0_round1_train_state" "${PYTHON_CMD[@]}" tools/run_rg_flux_iterative_inference.py \
    --checkpoint "${F0_ADAPTER}" --config "${CONFIG}" --input "${TRAIN_INPUT}" \
    --jsonl_path "${TRAIN_JSONL}" --output_dir "${RUN_ROOT}/01_f0_round1_train_state" \
    --iterations 1 --upscale 1 --seed "${SEED}" \
    --metric_device "${ITER_METRIC_DEVICE}" --metrics "${ITER_METRICS[@]}"

  run_stage "02_build_states_for_c" "${PYTHON_CMD[@]}" tools/build_sr_refinement_states.py \
    --lineage_jsonl "${RUN_ROOT}/01_f0_round1_train_state/sample_lineage.jsonl" \
    --source_jsonl "${TRAIN_JSONL}" --dataset_id "${DATASET_ID}" \
    --artifact_root "${RUN_ROOT}/01_f0_round1_train_state" --producer_adapter "${F0_ADAPTER}" \
    --output_jsonl "${RUN_ROOT}/02_states_for_c.jsonl" --max_round 2
}

run_c() {
  require_file "${RUN_ROOT}/02_states_for_c.jsonl"
  run_stage "03_shared_refiner_sft" "${PYTHON_CMD[@]}" train_rg_flux_refiner_sft.py \
    --config "${CONFIG}" --state_jsonl "${RUN_ROOT}/02_states_for_c.jsonl" \
    --init_adapter "${F0_ADAPTER}" --output_dir "${RUN_ROOT}/03_g_sft" \
    --max_steps "${SFT_MAX_STEPS}"
}

run_multiround() {
  local g_sft_adapter="${RUN_ROOT}/03_g_sft/rg_flux_adapters"
  require_file "${g_sft_adapter}"
  run_stage "04_sft_multiround_train_state" "${PYTHON_CMD[@]}" tools/run_rg_flux_iterative_inference.py \
    --checkpoint "${F0_ADAPTER}" --refiner_checkpoint "${g_sft_adapter}" \
    --config "${CONFIG}" --input "${TRAIN_INPUT}" --jsonl_path "${TRAIN_JSONL}" \
    --output_dir "${RUN_ROOT}/04_sft_multiround_train_state" \
    --iterations "${ITERATIONS}" --upscale 1 --seed "${SEED}" \
    --metric_device "${ITER_METRIC_DEVICE}" --metrics "${ITER_METRICS[@]}"

  run_stage "05_build_states_for_rl" "${PYTHON_CMD[@]}" tools/build_sr_refinement_states.py \
    --lineage_jsonl "${RUN_ROOT}/04_sft_multiround_train_state/sample_lineage.jsonl" \
    --source_jsonl "${TRAIN_JSONL}" --dataset_id "${DATASET_ID}" \
    --artifact_root "${RUN_ROOT}/04_sft_multiround_train_state" --producer_adapter "${g_sft_adapter}" \
    --round_producer_adapter "1=${F0_ADAPTER}" \
    --output_jsonl "${RUN_ROOT}/05_states_for_rl.jsonl" --max_round "${ITERATIONS}"

  # SFT evaluation is generated before reward/NFT. If evaluation.jsonl_path is
  # omitted, it is the training manifest, whose multiround metrics already exist.
  if [[ "${EVAL_INPUT}" == "${TRAIN_INPUT}" && "${EVAL_JSONL}" == "${TRAIN_JSONL}" ]]; then
    ln -sfn "04_sft_multiround_train_state" "${RUN_ROOT}/06_sft_multiround_eval"
  else
    run_stage "06_sft_multiround_eval" "${PYTHON_CMD[@]}" tools/run_rg_flux_iterative_inference.py \
      --checkpoint "${F0_ADAPTER}" --refiner_checkpoint "${g_sft_adapter}" \
      --config "${CONFIG}" --input "${EVAL_INPUT}" --jsonl_path "${EVAL_JSONL}" \
      --output_dir "${RUN_ROOT}/06_sft_multiround_eval" \
      --iterations "${ITERATIONS}" --upscale 1 --seed "${SEED}" \
      --metric_device "${ITER_METRIC_DEVICE}" --metrics "${ITER_METRICS[@]}"
  fi
}

run_reward() {
  local g_sft_adapter="${RUN_ROOT}/03_g_sft/rg_flux_adapters"
  require_file "${RUN_ROOT}/05_states_for_rl.jsonl"
  require_file "${g_sft_adapter}"
  run_stage "07_collect_rollouts" "${PYTHON_CMD[@]}" tools/collect_sr_output_rollouts.py \
    --config "${CONFIG}" --state_jsonl "${RUN_ROOT}/05_states_for_rl.jsonl" \
    --old_adapter "${g_sft_adapter}" --output_dir "${RUN_ROOT}/07_rollouts_gsft" \
    --num_candidates "${NUM_CANDIDATES}" --seed "${SEED}" --device "${REWARD_DEVICE}"

  run_stage "08_calibrate_reward" "${PYTHON_CMD[@]}" tools/calibrate_sr_reward.py \
    --rollout_jsonl "${RUN_ROOT}/07_rollouts_gsft/rollouts.jsonl" \
    --output_json "${RUN_ROOT}/08_reward_calibration.json" --device "${REWARD_DEVICE}"

  run_stage "09_score_rollouts" "${PYTHON_CMD[@]}" tools/score_sr_rollouts.py \
    --state_jsonl "${RUN_ROOT}/05_states_for_rl.jsonl" \
    --rollout_jsonl "${RUN_ROOT}/07_rollouts_gsft/rollouts.jsonl" \
    --calibration_json "${RUN_ROOT}/08_reward_calibration.json" \
    --output_jsonl "${RUN_ROOT}/09_scored_rollouts.jsonl" --device "${REWARD_DEVICE}"
}

run_e() {
  local g_sft_adapter="${RUN_ROOT}/03_g_sft/rg_flux_adapters"
  require_file "${RUN_ROOT}/09_scored_rollouts.jsonl"
  require_file "${g_sft_adapter}"
  run_stage "10_output_nft" "${PYTHON_CMD[@]}" train_rg_flux_output_rl.py \
    --config "${CONFIG}" --state_jsonl "${RUN_ROOT}/05_states_for_rl.jsonl" \
    --scored_rollout_jsonl "${RUN_ROOT}/09_scored_rollouts.jsonl" \
    --reference_adapter "${g_sft_adapter}" --old_adapter "${g_sft_adapter}" \
    --policy_adapter "${g_sft_adapter}" --output_dir "${RUN_ROOT}/10_g_rl_01" \
    --max_steps "${RL_MAX_STEPS}"
}

run_eval() {
  local g_rl_adapter="${RUN_ROOT}/10_g_rl_01/rg_flux_adapters"
  require_file "${g_rl_adapter}"
  run_stage "11_rl_multiround_eval" "${PYTHON_CMD[@]}" tools/run_rg_flux_iterative_inference.py \
    --checkpoint "${F0_ADAPTER}" --refiner_checkpoint "${g_rl_adapter}" \
    --config "${CONFIG}" --input "${EVAL_INPUT}" --jsonl_path "${EVAL_JSONL}" \
    --output_dir "${RUN_ROOT}/11_rl_multiround_eval" \
    --iterations "${ITERATIONS}" --upscale 1 --seed "${SEED}" \
    --metric_device "${ITER_METRIC_DEVICE}" --metrics "${ITER_METRICS[@]}"

  run_stage "12_summarize_sft_vs_rl" "${PYTHON_CMD[@]}" tools/summarize_rl_sr_evaluation.py \
    --sft_metric_trends "${RUN_ROOT}/06_sft_multiround_eval/metric_trends.csv" \
    --rl_metric_trends "${RUN_ROOT}/11_rl_multiround_eval/metric_trends.csv" \
    --output_json "${RUN_ROOT}/12_sft_to_rl_evaluation.json"
}

prepare_inputs
case "${STAGE}" in
  all) run_f0; run_c; run_multiround; run_reward; run_e; run_eval ;;
  f0) run_f0 ;;
  c) run_c ;;
  multiround) run_multiround ;;
  reward) run_reward ;;
  e) run_e ;;
  eval) run_eval ;;
  *)
    echo "Usage: $0 [all|f0|c|multiround|reward|e|eval]" >&2
    exit 2
    ;;
esac

echo "RL-SR stage ${STAGE} completed. Results and per-stage logs: ${RUN_ROOT}"
