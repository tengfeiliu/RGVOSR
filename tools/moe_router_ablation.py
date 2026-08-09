import csv
import hashlib
import inspect
import json
import math
import random
import types
from collections import defaultdict
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # Utility-only commands and tests can run without the training environment.
    torch = None


ROUTER_ABLATION_MODES = (
    "learned_top2",
    "fixed_mean",
    "shuffle_condition8",
    "uniform",
    "dense_soft",
    "onehot",
)


def _as_float_list(value):
    if value is None:
        return None
    if torch is not None and torch.is_tensor(value):
        value = value.detach().float().cpu()
        if value.ndim == 2:
            if value.shape[0] != 1:
                raise ValueError("Router ablation inference requires batch size one.")
            value = value[0]
        return [float(item) for item in value.reshape(-1).tolist()]
    return [float(item) for item in value]


def normalize_router_weights(weights, num_experts=None):
    if weights is None:
        raise ValueError("Router weights are required.")
    values = [float(value) for value in weights]
    if num_experts is not None and len(values) != int(num_experts):
        raise ValueError(
            f"Expected {num_experts} router weights, received {len(values)}."
        )
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("Router weights must be finite, non-negative values.")
    total = sum(values)
    if total <= 0.0:
        raise ValueError("Router weights must have a positive sum.")
    return [value / total for value in values]


def _checkpoint_step_number(value):
    text = str(value).strip()
    if text.startswith("checkpoint-"):
        text = text[len("checkpoint-") :]
    return int(text)


def mean_router_weights_from_history(
    history_path,
    checkpoint_step,
    last_n=1000,
    weight_field="usage",
    return_metadata=False,
):
    history_path = Path(history_path)
    if not history_path.exists():
        raise FileNotFoundError(f"Router history does not exist: {history_path}")
    checkpoint_step = _checkpoint_step_number(checkpoint_step)
    last_n = int(last_n)
    if last_n <= 0:
        raise ValueError("last_n must be positive.")

    with history_path.open("r", newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise ValueError(f"Router history is empty: {history_path}")

    if weight_field not in {"usage", "used"}:
        raise ValueError("weight_field must be 'usage' or 'used'.")
    suffix = f"_{weight_field}"
    expert_indexes = sorted(
        {
            int(key[len("router/expert_") : -len(suffix)])
            for key in raw_rows[0]
            if key.startswith("router/expert_") and key.endswith(suffix)
        }
    )
    if not expert_indexes or expert_indexes != list(range(len(expert_indexes))):
        raise ValueError(
            f"Could not discover contiguous expert_i_{weight_field} columns in {history_path}"
        )

    # Resume training appends to the CSV. Keep the latest row for duplicated steps.
    by_step = {}
    for row in raw_rows:
        raw_step = row.get("global_step")
        if raw_step in (None, ""):
            continue
        step = int(float(raw_step))
        if step <= checkpoint_step:
            by_step[step] = row
    selected_steps = sorted(by_step)[-last_n:]
    if not selected_steps:
        raise ValueError(
            f"No router history rows at or before checkpoint step {checkpoint_step}."
        )

    normalized_rows = []
    for step in selected_steps:
        row = by_step[step]
        values = []
        for index in expert_indexes:
            value = row.get(f"router/expert_{index}_{weight_field}")
            if value in (None, ""):
                raise ValueError(
                    f"Missing router/expert_{index}_{weight_field} at global_step={step}."
                )
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"Invalid router/expert_{index}_{weight_field}={value!r} "
                    f"at global_step={step}."
                )
            values.append(value)
        normalized_rows.append(normalize_router_weights(values, len(expert_indexes)))
    weights = normalize_router_weights(
        [
            sum(row[index] for row in normalized_rows) / len(normalized_rows)
            for index in expert_indexes
        ]
    )
    metadata = {
        "history_path": str(history_path),
        "weight_field": weight_field,
        "requested_last_n": last_n,
        "selected_step_start": selected_steps[0],
        "selected_step_end": selected_steps[-1],
        "selected_step_count": len(selected_steps),
        "weights": weights,
    }
    return (weights, metadata) if return_metadata else weights


def deterministic_derangement(size, seed):
    size = int(size)
    if size < 2:
        raise ValueError("Condition8 shuffling requires at least two samples per dataset.")
    order = list(range(size))
    random.Random(int(seed)).shuffle(order)
    # A cyclic shift of a shuffled order gives a one-to-one mapping without self-pairs.
    donors = [0] * size
    for position, source in enumerate(order):
        donors[source] = order[(position + 1) % size]
    if sorted(donors) != list(range(size)) or any(i == donor for i, donor in enumerate(donors)):
        raise AssertionError("Failed to construct a deterministic derangement.")
    return donors


def load_shuffled_condition_records(reference_trace_path, seed):
    reference_trace_path = Path(reference_trace_path)
    if not reference_trace_path.exists():
        raise FileNotFoundError(f"Reference route trace does not exist: {reference_trace_path}")
    first_steps = {}
    with reference_trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if int(row.get("step_index", -1)) == 0:
                first_steps[int(row["sample_index"])] = row
    if not first_steps:
        raise ValueError(f"Reference trace has no step_index=0 records: {reference_trace_path}")
    source_rows = [first_steps[index] for index in sorted(first_steps)]
    if [int(row["sample_index"]) for row in source_rows] != list(range(len(source_rows))):
        raise ValueError("Reference trace sample indexes must be contiguous from zero.")

    dataset_groups = defaultdict(list)
    for index, row in enumerate(source_rows):
        dataset_groups[str(row.get("dataset") or "default")].append(index)

    shuffled = [None] * len(source_rows)
    for dataset_offset, (dataset, indexes) in enumerate(sorted(dataset_groups.items())):
        donors = deterministic_derangement(len(indexes), int(seed) + dataset_offset * 1009)
        for local_source, local_donor in enumerate(donors):
            source_index = indexes[local_source]
            donor_index = indexes[local_donor]
            donor = source_rows[donor_index]
            if donor.get("router_condition") is None:
                raise ValueError(
                    f"Reference trace sample {donor_index} has no router_condition."
                )
            shuffled[source_index] = {
                "source_router_condition": source_rows[source_index]["router_condition"],
                "source_router_condition_mask": source_rows[source_index].get(
                    "router_condition_mask"
                ),
                "source_router_condition_confidence": source_rows[source_index].get(
                    "router_condition_confidence"
                ),
                "router_condition": donor["router_condition"],
                "router_condition_mask": donor.get("router_condition_mask"),
                "router_condition_confidence": donor.get("router_condition_confidence"),
                "donor_sample_index": donor_index,
                "donor_dataset": dataset,
                "donor_source_image_path": donor.get("source_image_path"),
            }
    return shuffled


def checkpoint_fingerprint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    digest = hashlib.sha256()
    files = sorted(path for path in checkpoint_dir.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(checkpoint_dir).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return {"sha256": digest.hexdigest(), "file_count": len(files)}


class RouterAblationController:
    """Process-local Router intervention installed only by the diagnostic entrypoint."""

    def __init__(
        self,
        mode,
        num_steps,
        fixed_weights=None,
        onehot_expert=None,
        shuffle_reference_trace=None,
        shuffle_seed=3407,
    ):
        if mode not in ROUTER_ABLATION_MODES:
            raise ValueError(f"Unsupported Router ablation mode: {mode}")
        self.mode = mode
        self.num_steps = int(num_steps)
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        self.fixed_weights = fixed_weights
        self.onehot_expert = onehot_expert
        self.shuffle_records = None
        if mode == "shuffle_condition8":
            if not shuffle_reference_trace:
                raise ValueError("shuffle_condition8 requires --shuffle_reference_trace.")
            self.shuffle_records = load_shuffled_condition_records(
                shuffle_reference_trace, seed=shuffle_seed
            )
        self.records = []
        self._call_index = 0
        self._router = None
        self._original_forward = None
        self._num_experts = None
        self._routing_schedule = None

    @staticmethod
    def _condition_hash(condition, condition_mask, condition_confidence):
        if condition is None:
            return None
        payload = json.dumps(
            {
                "condition": condition,
                "mask": condition_mask,
                "confidence": condition_confidence,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _active_indexes(values):
        return [
            index
            for index, value in sorted(
                enumerate(values), key=lambda item: (-item[1], item[0])
            )
            if value > 0.0
        ]

    def _override_weights(self, router_output):
        if torch is None:
            raise RuntimeError("PyTorch is required for Router ablation inference.")
        alpha = router_output["alpha"]
        if alpha.ndim != 2 or alpha.shape[0] != 1:
            raise ValueError("Router ablation inference requires alpha shape [1, E].")
        num_experts = alpha.shape[-1]
        if self.mode in {"learned_top2", "shuffle_condition8"}:
            return alpha
        if self.mode == "dense_soft":
            return router_output["clean_dense_alpha"]
        if self.mode == "uniform":
            weights = [1.0 / num_experts] * num_experts
        elif self.mode == "fixed_mean":
            weights = normalize_router_weights(self.fixed_weights, num_experts=num_experts)
        elif self.mode == "onehot":
            if self.onehot_expert is None or not 0 <= int(self.onehot_expert) < num_experts:
                raise ValueError(f"onehot_expert must be in [0, {num_experts - 1}].")
            weights = [0.0] * num_experts
            weights[int(self.onehot_expert)] = 1.0
        else:
            raise AssertionError(self.mode)
        return torch.tensor(weights, device=alpha.device, dtype=alpha.dtype).unsqueeze(0)

    @staticmethod
    def _tensor_from_record(values, reference, reshape_confidence=False):
        if values is None:
            return None
        if torch is None:
            raise RuntimeError("PyTorch is required for Router ablation inference.")
        tensor = torch.tensor(values, device=reference.device, dtype=reference.dtype)
        if reshape_confidence:
            return tensor.reshape(-1)
        return tensor.reshape(1, -1)

    def _install_shuffle_condition(self, bound, sample_index):
        if self.shuffle_records is None:
            return None
        if sample_index >= len(self.shuffle_records):
            raise ValueError(
                "Current inference contains more samples than the learned_top2 reference trace."
            )
        donor = self.shuffle_records[sample_index]
        reference = bound.arguments.get("router_condition")
        if reference is None:
            raise ValueError("shuffle_condition8 requires a condition8 Router input.")
        bound.arguments["router_condition"] = self._tensor_from_record(
            donor["router_condition"], reference
        )
        mask = donor.get("router_condition_mask")
        bound.arguments["router_condition_mask"] = self._tensor_from_record(mask, reference)
        confidence = donor.get("router_condition_confidence")
        bound.arguments["router_condition_confidence"] = self._tensor_from_record(
            confidence, reference, reshape_confidence=True
        )
        return donor

    def install(self, artist):
        router = getattr(artist, "moe_router", None)
        if router is None:
            raise ValueError("Router ablation requires a LoRA-MoE checkpoint with moe_router.")
        if self._router is not None:
            raise RuntimeError("Router ablation controller is already installed.")
        self._router = router
        self._original_forward = router.forward
        signature = inspect.signature(self._original_forward)
        self._num_experts = int(getattr(router, "num_experts"))
        controller = self

        def wrapped_forward(_router, *args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs)
            routing_mode = str(bound.arguments.get("routing_mode", "soft"))
            top_k = int(bound.arguments.get("top_k", 2))
            temperature = float(bound.arguments.get("temperature", 1.0))
            if routing_mode != "topk" or top_k != 2:
                raise ValueError(
                    "The learned_top2 ablation baseline requires inference routing_mode='topk' "
                    f"and top_k=2, received routing_mode={routing_mode!r}, top_k={top_k}."
                )
            schedule = {
                "routing_mode": routing_mode,
                "top_k": top_k,
                "temperature": temperature,
            }
            if controller._routing_schedule is None:
                controller._routing_schedule = schedule
            elif controller._routing_schedule != schedule:
                raise ValueError("MoE inference routing schedule changed within one ablation run.")
            sample_index = controller._call_index // controller.num_steps
            step_index = controller._call_index % controller.num_steps
            donor = None
            if controller.mode == "shuffle_condition8":
                donor = controller._install_shuffle_condition(bound, sample_index)
            result = controller._original_forward(*bound.args, **bound.kwargs)
            if not isinstance(result, dict) or "alpha" not in result:
                raise ValueError(
                    "Router ablation requires the MoE caller to request return_details=True."
                )
            overridden = dict(result)
            overridden["alpha"] = controller._override_weights(result)

            condition = bound.arguments.get("router_condition")
            condition_mask = bound.arguments.get("router_condition_mask")
            condition_confidence = bound.arguments.get("router_condition_confidence")
            timestep = bound.arguments.get("timestep")
            condition_values = _as_float_list(condition)
            condition_mask_values = _as_float_list(condition_mask)
            condition_confidence_values = _as_float_list(condition_confidence)
            learned_values = _as_float_list(result["alpha"])
            used_values = _as_float_list(overridden["alpha"])
            record = {
                "sample_index": sample_index,
                "step_index": step_index,
                "timestep": (_as_float_list(timestep) or [None])[0],
                "mode": controller.mode,
                "routing_mode": routing_mode,
                "top_k": top_k,
                "temperature": temperature,
                "router_condition": condition_values,
                "router_condition_mask": condition_mask_values,
                "router_condition_confidence": condition_confidence_values,
                "router_condition_hash": controller._condition_hash(
                    condition_values,
                    condition_mask_values,
                    condition_confidence_values,
                ),
                "dense_alpha": _as_float_list(result.get("clean_dense_alpha")),
                "learned_alpha": learned_values,
                "learned_top_indices": controller._active_indexes(learned_values),
                "used_alpha": used_values,
                "used_top_indices": controller._active_indexes(used_values),
            }
            if donor is not None:
                source_hash = controller._condition_hash(
                    donor["source_router_condition"],
                    donor["source_router_condition_mask"],
                    donor["source_router_condition_confidence"],
                )
                record.update(
                    {
                        "source_router_condition_hash": source_hash,
                        "condition_donor_sample_index": donor["donor_sample_index"],
                        "condition_donor_dataset": donor["donor_dataset"],
                        "condition_donor_source_image_path": donor[
                            "donor_source_image_path"
                        ],
                    }
                )
            controller.records.append(record)
            controller._call_index += 1
            return overridden

        router.forward = types.MethodType(wrapped_forward, router)
        return self

    def uninstall(self):
        if self._router is not None and self._original_forward is not None:
            self._router.forward = self._original_forward
        self._router = None
        self._original_forward = None

    def _sample_metadata(self, inference_manifest_path):
        inference_manifest_path = Path(inference_manifest_path)
        with inference_manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        metadata = []
        for dataset in manifest.get("datasets", []):
            pairing_path = dataset.get("iqa_pairing_manifest") or dataset.get(
                "suggestion_pairing_manifest"
            )
            if not pairing_path:
                raise ValueError(
                    f"Inference manifest dataset has no pairing manifest: {dataset.get('name')}"
                )
            with Path(pairing_path).open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    metadata.append(
                        {
                            "dataset": dataset.get("name"),
                            "source_image_path": row.get("source_image_path"),
                            "source_lq_path": row.get("source_lq_path"),
                            "output_image_path": row.get("output_image_path"),
                        }
                    )
        return metadata, manifest

    def finalize(self, trace_path, inference_manifest_path):
        if self._call_index % self.num_steps != 0:
            raise ValueError(
                f"Router call count {self._call_index} is not divisible by num_steps={self.num_steps}."
            )
        sample_count = self._call_index // self.num_steps
        metadata, manifest = self._sample_metadata(inference_manifest_path)
        if len(metadata) != sample_count:
            raise ValueError(
                f"Trace has {sample_count} samples but inference manifests contain {len(metadata)}."
            )
        if self.shuffle_records is not None and len(self.shuffle_records) != sample_count:
            raise ValueError(
                "shuffle_condition8 must use the same datasets and image ordering as learned_top2."
            )
        for record in self.records:
            record.update(metadata[int(record["sample_index"])])

        shuffle_summary = None
        if self.shuffle_records is not None:
            first_steps = [record for record in self.records if record["step_index"] == 0]
            self_pairs = sum(
                int(record["sample_index"]) == int(record["condition_donor_sample_index"])
                for record in first_steps
            )
            same_conditions = sum(
                record.get("source_router_condition_hash") == record.get("router_condition_hash")
                for record in first_steps
            )
            shuffle_summary = {
                "self_pair_count": self_pairs,
                "same_condition_count": same_conditions,
                "same_condition_ratio": same_conditions / max(len(first_steps), 1),
            }

        trace_path = Path(trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {
            "mode": self.mode,
            "trace_path": str(trace_path),
            "sample_count": sample_count,
            "router_call_count": self._call_index,
            "num_inference_steps": self.num_steps,
            "num_experts": self._num_experts,
            "fixed_weights": (
                normalize_router_weights(self.fixed_weights, self._num_experts)
                if self.mode == "fixed_mean"
                else None
            ),
            "onehot_expert": int(self.onehot_expert) if self.mode == "onehot" else None,
            "learned_router_schedule": self._routing_schedule,
            "shuffle_summary": shuffle_summary,
            "base_inference_manifest": str(inference_manifest_path),
            "datasets": [item.get("name") for item in manifest.get("datasets", [])],
        }
