#!/usr/bin/env bash
# RL-SR A--E 专用后台命令入口。
#
# 推荐用法（从仓库根目录执行）：
#   F0_CHECKPOINT=/path/to/checkpoint-00036000 \
#   F0_REUSE_DIR=/path/to/old_run/01_f0_round1_train_state \
#   bash commands/rl_sr_stage_ae_commands.sh reuse-all
#
# 命令与阶段对应关系：
#   reuse-all       A 配置 -> B(00/复用01/02) -> C(03/04/05/06) -> D(07/08/09)
#                   -> E(10) -> 最终评估(11/12)。01 不重新推理或算 IQA，04 的 y_1
#                   也复用；04 只从第 2 轮开始生成，并对所选样本计算逐轮 IQA。
#   fresh-all       与 reuse-all 相同，但 01 和 04 的第 1 轮都重新用 F0 生成。
#   resume-from04   当前04已生成部分/全部轮次并卡在训练集IQA时使用：验证并复用现有
#                   04图片，从下一未生成轮次继续，跳过04后续IQA，然后自动完成05--12。
#   resume-from05   04已完成、05或其后失败时使用：不再运行04，直接重建05并自动完成
#                   06--12。已有完整阶段输出可由各工具自身的恢复逻辑复用。
#   resume-from06   05已完成、06推理或指标失败时使用：复用06现有轮次图片，补算缺失
#                   指标并自动完成07--12。
#   resume-c        只运行 03_shared_refiner_sft；要求 02_states_for_c.jsonl 已存在。
#   resume-multiround
#                   运行 04_sft_multiround_train_state、05_build_states_for_rl、
#                   06_sft_multiround_evaluation；要求 03_g_sft 已存在。
#   resume-reward   D 阶段：07_collect_rollouts、08_calibrate_reward、09_score_rollouts。
#   resume-e        E 阶段：10_output_nft。
#   resume-eval     最终评估：11_rl_multiround_evaluation、12_summarize_sft_vs_rl。
#   inspect          不启动任务；列出已有运行目录中各阶段、图片和性能指标的完成情况。
#
# 所有模式最终调用 tools/run_rl_sr_stage_ae.sh。训练与评估 JSONL 均从 CONFIG 读取，
# 不需要 LQ_ROOT、DATASET_ROOT 或额外拆分 JSONL。每次新实验自动创建带配置与时间的
# 输出目录；启动器自己的控制台日志写入 launcher_logs/，各步骤详细日志写入实验目录 logs/。
# 01/04 训练状态沿用配置中的 512x512 pre-cropped 输入；06/11 验证固定使用
# full-frame 推理并恢复每张输入图的原始尺寸，不对 RealLQ250/RealLR200 做 512 裁剪。

set -Eeuo pipefail

ACTION="${1:-help}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/rl_sr_refinement_flux2_klein_moe.yaml}"
CONDA_ENV="${CONDA_ENV:-sr-flux2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  if [[ -n "${OMP_NUM_THREADS:-}" ]]; then
    echo "Warning: invalid OMP_NUM_THREADS='${OMP_NUM_THREADS}'; using 1." >&2
  fi
  OMP_NUM_THREADS=1
fi
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-4000}"
TRAIN_SUBSET_SEED="${TRAIN_SUBSET_SEED:-42}"
F0_TRAIN_IQA_SAMPLES="${F0_TRAIN_IQA_SAMPLES:-0}"
MULTIROUND_TRAIN_IQA_SAMPLES="${MULTIROUND_TRAIN_IQA_SAMPLES:-0}"
COMMAND_LOG_ROOT="${COMMAND_LOG_ROOT:-${REPO_ROOT}/launcher_logs}"
RUNNER="${REPO_ROOT}/tools/run_rl_sr_stage_ae.sh"

usage() {
  cat <<'EOF'
Usage:
  F0_CHECKPOINT=<checkpoint> F0_REUSE_DIR=<old-01-dir> \
    bash commands/rl_sr_stage_ae_commands.sh reuse-all

  F0_CHECKPOINT=<checkpoint> \
    bash commands/rl_sr_stage_ae_commands.sh fresh-all

  RL_SR_RUN_DIR=<existing-run-dir> [F0_CHECKPOINT=<checkpoint>] \
    bash commands/rl_sr_stage_ae_commands.sh \
    {resume-from04|resume-from05|resume-from06|resume-c|resume-multiround|resume-reward|resume-e|resume-eval}

  RL_SR_RUN_DIR=<existing-run-dir> \
    bash commands/rl_sr_stage_ae_commands.sh inspect

Optional environment variables:
  CONFIG                 YAML 配置，默认 configs/rl_sr_refinement_flux2_klein_moe.yaml
  TRAIN_MAX_SAMPLES      新实验训练样本上限，默认 4000；0 表示全部
  TRAIN_SUBSET_SEED      固定子集种子，默认 42
  F0_TRAIN_IQA_SAMPLES   01训练集IQA：-1=全量，0=跳过，正整数=抽样；默认0
  MULTIROUND_TRAIN_IQA_SAMPLES
                         04每轮训练集IQA：-1=全量，0=跳过，正整数=抽样；默认0
  CUDA_VISIBLE_DEVICES   使用的 GPU，默认 0
  OMP_NUM_THREADS        CPU OpenMP 线程数，必须为正整数；非法或未设置时使用 1
  CONDA_ENV              Conda 环境，默认 sr-flux2
  RL_SR_OUTPUT_ROOT      实验输出根目录
  COMMAND_LOG_ROOT       后台启动日志目录，默认 launcher_logs

说明：
  reuse-all/fresh-all 会创建新实验，因此忽略 RL_SR_RUN_DIR。
  resume-* 必须提供 RL_SR_RUN_DIR，并继承该实验保存的样本数量与抽样种子。
  inspect 只检查文件，不需要 F0_CHECKPOINT，也不会启动训练或评估。
  resume-* 会优先验证 F0_CHECKPOINT；无效或未设置时，从现有04的
  iterative_manifest.json 自动恢复实际 checkpoint_path。
EOF
}

require_value() {
  local name="$1"
  local value="$2"
  if [[ -z "${value}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    usage >&2
    exit 2
  fi
}

require_path() {
  local name="$1"
  local value="$2"
  require_value "${name}" "${value}"
  if [[ ! -e "${value}" ]]; then
    echo "${name} does not exist: ${value}" >&2
    exit 2
  fi
}

is_f0_adapter() {
  local candidate="$1"
  [[ -f "${candidate}/flux2_klein_lora_moe_state.pt" ]] || \
    [[ -f "${candidate}/rg_flux_adapters/flux2_klein_lora_moe_state.pt" ]]
}

resolve_resume_f0_checkpoint() {
  if [[ -n "${F0_CHECKPOINT:-}" ]] && is_f0_adapter "${F0_CHECKPOINT}"; then
    return
  fi

  local manifest="${RL_SR_RUN_DIR}/04_sft_multiround_train_state/iterative_manifest.json"
  if [[ -f "${manifest}" ]]; then
    local discovered
    discovered="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("checkpoint_path", ""))' "${manifest}")"
    if [[ -n "${discovered}" ]] && is_f0_adapter "${discovered}"; then
      if [[ -n "${F0_CHECKPOINT:-}" ]]; then
        printf 'Ignoring invalid F0_CHECKPOINT=%s\n' "${F0_CHECKPOINT}" >&2
      fi
      F0_CHECKPOINT="${discovered}"
      export F0_CHECKPOINT
      printf 'Recovered F0 checkpoint from 04 manifest: %s\n' "${F0_CHECKPOINT}"
      return
    fi
  fi

  echo "Cannot resolve a valid F0 adapter. Set F0_CHECKPOINT to checkpoint-00036000 (not 01_f0_round1_train_state)." >&2
  exit 2
}

launch_new() {
  local label="$1"
  local reuse_dir="$2"
  local timestamp
  timestamp="$(date +%y%m%d-%H%M%S)"
  mkdir -p "${COMMAND_LOG_ROOT}"
  local log_path="${COMMAND_LOG_ROOT}/rlsr_${label}_n${TRAIN_MAX_SAMPLES}_ss${TRAIN_SUBSET_SEED}_${timestamp}.log"

  nohup env -u RL_SR_RUN_DIR \
    REPO_ROOT="${REPO_ROOT}" CONFIG="${CONFIG}" F0_CHECKPOINT="${F0_CHECKPOINT}" \
    F0_REUSE_DIR="${reuse_dir}" \
    TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES}" TRAIN_SUBSET_SEED="${TRAIN_SUBSET_SEED}" \
    F0_TRAIN_IQA_SAMPLES="${F0_TRAIN_IQA_SAMPLES}" \
    MULTIROUND_TRAIN_IQA_SAMPLES="${MULTIROUND_TRAIN_IQA_SAMPLES}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
    TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CONDA_ENV="${CONDA_ENV}" \
    bash "${RUNNER}" all \
    > "${log_path}" 2>&1 < /dev/null &
  local pid=$!
  printf 'Started PID %s\nLauncher log: %s\n' "${pid}" "${log_path}"
  printf 'The generated experiment directory will be printed near the beginning of that log.\n'
}

launch_resume() {
  local runner_stage="$1"
  local label="$2"
  local timestamp
  timestamp="$(date +%y%m%d-%H%M%S)"
  mkdir -p "${COMMAND_LOG_ROOT}"
  local log_path="${COMMAND_LOG_ROOT}/rlsr_${label}_${timestamp}.log"

  nohup env \
    REPO_ROOT="${REPO_ROOT}" CONFIG="${CONFIG}" F0_CHECKPOINT="${F0_CHECKPOINT}" \
    RL_SR_RUN_DIR="${RL_SR_RUN_DIR}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
    TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CONDA_ENV="${CONDA_ENV}" \
    bash "${RUNNER}" "${runner_stage}" \
    > "${log_path}" 2>&1 < /dev/null &
  local pid=$!
  printf 'Started PID %s\nExisting experiment: %s\nLauncher log: %s\n' \
    "${pid}" "${RL_SR_RUN_DIR}" "${log_path}"
}

inspect_results() {
  local run_root="$1"
  printf 'RL-SR experiment: %s\n\n' "${run_root}"

  local entries=(
    "B/01 y_1 images and lineage|01_f0_round1_train_state/sample_lineage.jsonl"
    "B/02 states for C|02_states_for_c.jsonl"
    "C/03 G_SFT adapter|03_g_sft/rg_flux_adapters"
    "C/03 training summary|03_g_sft/summary.json"
    "C/04 training-set multiround metrics|04_sft_multiround_train_state/metric_trends.csv"
    "C/05 states for RL|05_states_for_rl.jsonl"
    "C/06 G_SFT evaluation metrics|06_sft_multiround_evaluation/metric_trends.csv"
    "D/07 rollout manifest|07_rollouts_gsft/rollout_manifest.json"
    "D/08 reward calibration|08_reward_calibration.json"
    "D/09 scored reward summary|09_scored_rollouts.summary.json"
    "E/10 G_RL adapter|10_g_rl_01/rg_flux_adapters"
    "E/10 training summary|10_g_rl_01/summary.json"
    "Evaluation/11 G_RL metrics|11_rl_multiround_evaluation/metric_trends.csv"
    "Evaluation/12 G_SFT vs G_RL|12_sft_to_rl_evaluation.json"
  )
  local entry label relative target
  for entry in "${entries[@]}"; do
    label="${entry%%|*}"
    relative="${entry#*|}"
    target="${run_root}/${relative}"
    if [[ -e "${target}" ]]; then
      printf '[READY]   %-40s %s\n' "${label}" "${target}"
    else
      printf '[MISSING] %-40s %s\n' "${label}" "${target}"
    fi
  done

  printf '\nPer-round metric files already available:\n'
  local found=0
  local metric_file
  for metric_file in \
    "${run_root}"/04_sft_multiround_train_state/round_*/metrics/summary_scores.csv \
    "${run_root}"/06_sft_multiround_evaluation/round_*/metrics/summary_scores.csv \
    "${run_root}"/11_rl_multiround_evaluation/round_*/metrics/summary_scores.csv; do
    if [[ -f "${metric_file}" ]]; then
      printf '  %s\n' "${metric_file}"
      found=1
    fi
  done
  if [[ "${found}" -eq 0 ]]; then
    printf '  none (the pipeline has not completed a metric-evaluation round yet)\n'
  fi

  printf '\nStage logs:\n  %s\n' "${run_root}/logs"
}

if [[ "${ACTION}" == "help" || "${ACTION}" == "--help" || "${ACTION}" == "-h" ]]; then
  usage
  exit 0
fi

require_path "CONFIG" "${CONFIG}"
require_path "runner" "${RUNNER}"

case "${ACTION}" in
  reuse-all)
    require_path "F0_CHECKPOINT" "${F0_CHECKPOINT:-}"
    require_path "F0_REUSE_DIR" "${F0_REUSE_DIR:-}"
    launch_new "reuse_f0" "${F0_REUSE_DIR}"
    ;;
  fresh-all)
    require_path "F0_CHECKPOINT" "${F0_CHECKPOINT:-}"
    launch_new "fresh_f0" ""
    ;;
  resume-from04)
    require_path "RL_SR_RUN_DIR" "${RL_SR_RUN_DIR:-}"
    resolve_resume_f0_checkpoint
    launch_resume "from04" "resume_from04_to_final"
    ;;
  resume-from05)
    require_path "RL_SR_RUN_DIR" "${RL_SR_RUN_DIR:-}"
    resolve_resume_f0_checkpoint
    launch_resume "from05" "resume_from05_to_final"
    ;;
  resume-from06)
    require_path "RL_SR_RUN_DIR" "${RL_SR_RUN_DIR:-}"
    resolve_resume_f0_checkpoint
    launch_resume "from06" "resume_from06_to_final"
    ;;
  resume-c)
    require_path "RL_SR_RUN_DIR" "${RL_SR_RUN_DIR:-}"
    resolve_resume_f0_checkpoint
    launch_resume "c" "resume_c"
    ;;
  resume-multiround)
    require_path "RL_SR_RUN_DIR" "${RL_SR_RUN_DIR:-}"
    resolve_resume_f0_checkpoint
    launch_resume "multiround" "resume_multiround"
    ;;
  resume-reward)
    require_path "RL_SR_RUN_DIR" "${RL_SR_RUN_DIR:-}"
    resolve_resume_f0_checkpoint
    launch_resume "reward" "resume_reward"
    ;;
  resume-e)
    require_path "RL_SR_RUN_DIR" "${RL_SR_RUN_DIR:-}"
    resolve_resume_f0_checkpoint
    launch_resume "e" "resume_e"
    ;;
  resume-eval)
    require_path "RL_SR_RUN_DIR" "${RL_SR_RUN_DIR:-}"
    resolve_resume_f0_checkpoint
    launch_resume "eval" "resume_eval"
    ;;
  inspect)
    require_path "RL_SR_RUN_DIR" "${RL_SR_RUN_DIR:-}"
    inspect_results "$(cd "${RL_SR_RUN_DIR}" && pwd)"
    ;;
  *)
    echo "Unknown action: ${ACTION}" >&2
    usage >&2
    exit 2
    ;;
esac
