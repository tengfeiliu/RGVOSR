import hashlib
import json
import os
from pathlib import Path

import torch


TEXT_EMBEDDING_KEYS = ("prompt_embeds", "pooled_prompt_embeds", "text_ids")
PROMPT_BUILDER_SIGNATURE = "rg_flux_sr_prompt_builder_v2_caption_iqa"


def cfg(config, path, default=None):
    current = config
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def normalize_text_encoding_mode(config):
    mode = str(cfg(config, "text_encoding.mode", "online") or "online").strip().lower()
    if mode not in {"online", "cached", "auto"}:
        raise ValueError(f"Unsupported text_encoding.mode: {mode}")
    return mode


def text_encoder_should_load(config):
    return normalize_text_encoding_mode(config) != "cached"


def normalize_dtype_name(dtype):
    if dtype is None:
        return None
    if isinstance(dtype, str):
        return dtype.strip().lower()
    if dtype is torch.float16:
        return "fp16"
    if dtype is torch.bfloat16:
        return "bf16"
    if dtype is torch.float32:
        return "fp32"
    return str(dtype).replace("torch.", "").lower()


def normalize_image_key(value):
    if value is None:
        return ""
    return str(value).replace("\\", "/")


def stable_json_hash(payload):
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_prompt_hash(prompt):
    return hashlib.sha256(str(prompt or "").encode("utf-8")).hexdigest()


def compute_encoder_signature(config, dtype=None):
    payload = {
        "flux_backend": str(cfg(config, "model.flux_backend", "flux1") or "flux1").lower(),
        "flux_model_path": str(cfg(config, "model.flux_model_path", cfg(config, "flux_model_path", "")) or ""),
        "max_prompt_sequence_length": int(cfg(config, "model.max_prompt_sequence_length", 128) or 0),
        "text_encoder_dtype": normalize_dtype_name(cfg(config, "model.text_encoder_dtype", "fp32")),
        "use_prompt": bool(cfg(config, "condition.use_prompt", True)),
        "use_suggestions": bool(cfg(config, "condition.use_suggestions", True)),
        "include_caption": bool(cfg(config, "condition.include_caption", False)),
        "dtype": normalize_dtype_name(dtype or cfg(config, "text_encoding.dtype", cfg(config, "model.dtype", "bf16"))),
        "prompt_builder": PROMPT_BUILDER_SIGNATURE,
    }
    prompt_variant = cfg(config, "condition.prompt_variant", None)
    if prompt_variant is not None:
        payload["prompt_variant"] = str(prompt_variant)
    prompt_schedule = cfg(config, "condition.prompt_schedule", None)
    if isinstance(prompt_schedule, dict):
        payload["prompt_schedule"] = {
            "enabled": bool(prompt_schedule.get("enabled", False)),
            "switch_step": int(prompt_schedule.get("switch_step", 0) or 0),
            "before_variant": prompt_schedule.get("before_variant"),
            "before_include_caption": bool(
                prompt_schedule.get("before_include_caption", False)
            ),
            "after_variant": prompt_schedule.get("after_variant"),
            "after_include_caption": bool(
                prompt_schedule.get("after_include_caption", False)
            ),
        }
    return stable_json_hash(payload)


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _ensure_batch_dim(tensor, expected_min_dims):
    if tensor.ndim == expected_min_dims - 1:
        return tensor.unsqueeze(0)
    return tensor


def _move_embedding_state(state, device, dtype):
    prompt_embeds = _ensure_batch_dim(state["prompt_embeds"], 3).to(device=device, dtype=dtype)
    pooled_prompt_embeds = state["pooled_prompt_embeds"]
    if pooled_prompt_embeds is None:
        pooled_prompt_embeds = prompt_embeds.new_zeros(prompt_embeds.shape[0], 0)
    else:
        pooled_prompt_embeds = _ensure_batch_dim(pooled_prompt_embeds, 2).to(device=device, dtype=dtype)
    text_ids = state["text_ids"].to(device=device, dtype=dtype)
    return {
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "text_ids": text_ids,
    }


def stack_embedding_states(states, device, dtype):
    moved = [_move_embedding_state(state, device=device, dtype=dtype) for state in states]
    prompt_embeds = torch.cat([state["prompt_embeds"] for state in moved], dim=0)
    pooled_prompt_embeds = torch.cat([state["pooled_prompt_embeds"] for state in moved], dim=0)
    text_ids_list = [state["text_ids"] for state in moved]
    if text_ids_list[0].ndim >= 3:
        text_ids = torch.cat(text_ids_list, dim=0)
    else:
        text_ids = text_ids_list[0]
    return prompt_embeds, pooled_prompt_embeds, text_ids


class TextEmbeddingCache:
    def __init__(
        self,
        cache_dir,
        config,
        dtype=None,
        strict=True,
        validate_prompt_hash=True,
        load_existing=True,
    ):
        if not cache_dir:
            raise ValueError("text_encoding.cache_dir is required for cached text encoding.")
        self.cache_dir = Path(cache_dir)
        self.config = config
        self.dtype_name = normalize_dtype_name(dtype or cfg(config, "text_encoding.dtype", cfg(config, "model.dtype", "bf16")))
        self.strict = bool(strict)
        self.validate_prompt_hash = bool(validate_prompt_hash)
        self.encoder_signature = compute_encoder_signature(config, dtype=self.dtype_name)
        self.manifest_path = self.cache_dir / "manifest.jsonl"
        self.records_by_image = {}
        self.records_by_prompt = {}
        if load_existing:
            self.load_manifest()

    @classmethod
    def from_config(cls, config, dtype=None, load_existing=True):
        cache_dir = cfg(config, "text_encoding.cache_dir", None)
        if not cache_dir:
            return None
        return cls(
            cache_dir=cache_dir,
            config=config,
            dtype=dtype,
            strict=bool(cfg(config, "text_encoding.strict", True)),
            validate_prompt_hash=bool(cfg(config, "text_encoding.validate_prompt_hash", True)),
            load_existing=load_existing,
        )

    def load_manifest(self):
        self.records_by_image.clear()
        self.records_by_prompt.clear()
        if not self.manifest_path.exists():
            return
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                image_key = normalize_image_key(record.get("image_key"))
                prompt_hash = record.get("prompt_hash")
                encoder_signature = record.get("encoder_signature")
                if image_key:
                    self.records_by_image[image_key] = record
                if prompt_hash and encoder_signature:
                    self.records_by_prompt[(prompt_hash, encoder_signature)] = record

    def embedding_relative_path(self, prompt_hash):
        filename = f"{prompt_hash}_{self.encoder_signature}.pt"
        return Path("embeddings") / prompt_hash[:2] / prompt_hash[2:4] / filename

    def embedding_path(self, record_or_rel_path):
        rel_path = record_or_rel_path.get("embedding_path") if isinstance(record_or_rel_path, dict) else record_or_rel_path
        return self.cache_dir / Path(rel_path)

    def load_embedding_file(self, record):
        path = self.embedding_path(record)
        state = _torch_load(path)
        if not isinstance(state, dict):
            raise RuntimeError(f"Text embedding cache file is not a dict: {path}")
        missing = [key for key in TEXT_EMBEDDING_KEYS if key not in state]
        if missing:
            raise RuntimeError(f"Text embedding cache file {path} is missing keys: {missing}")
        for key in TEXT_EMBEDDING_KEYS:
            if not torch.is_tensor(state[key]):
                raise RuntimeError(f"Text embedding cache field {key} is not a tensor: {path}")
        if state["prompt_embeds"].ndim not in {2, 3}:
            raise RuntimeError(f"Invalid prompt_embeds shape in text embedding cache: {path}")
        if state["pooled_prompt_embeds"].ndim not in {1, 2}:
            raise RuntimeError(f"Invalid pooled_prompt_embeds shape in text embedding cache: {path}")
        if state["text_ids"].ndim not in {2, 3}:
            raise RuntimeError(f"Invalid text_ids shape in text embedding cache: {path}")
        if state.get("prompt_hash") not in {None, record.get("prompt_hash")}:
            raise RuntimeError(f"prompt_hash mismatch inside text embedding cache file: {path}")
        if state.get("encoder_signature") not in {None, record.get("encoder_signature")}:
            raise RuntimeError(f"encoder_signature mismatch inside text embedding cache file: {path}")
        return state

    def validate_record(self, record, prompt, image_key=None, validate_file=True):
        if not isinstance(record, dict):
            return False
        current_prompt_hash = compute_prompt_hash(prompt)
        if self.validate_prompt_hash and record.get("prompt_hash") != current_prompt_hash:
            return False
        if record.get("encoder_signature") != self.encoder_signature:
            return False
        if image_key and record.get("image_key") and normalize_image_key(record.get("image_key")) != normalize_image_key(image_key):
            return False
        if not self.embedding_path(record).exists():
            return False
        if validate_file:
            try:
                self.load_embedding_file(record)
            except Exception:
                return False
        return True

    def find_record(self, prompt, image_key=None, allow_prompt_reuse=True, validate_file=True):
        normalized_key = normalize_image_key(image_key)
        image_record = self.records_by_image.get(normalized_key)
        if image_record is not None and self.validate_record(
            image_record,
            prompt,
            normalized_key,
            validate_file=validate_file,
        ):
            return image_record
        if allow_prompt_reuse:
            prompt_hash = compute_prompt_hash(prompt)
            prompt_record = self.records_by_prompt.get((prompt_hash, self.encoder_signature))
            if prompt_record is not None and self.validate_record(
                prompt_record,
                prompt,
                validate_file=validate_file,
            ):
                return prompt_record
        return None

    def load_embedding(self, prompt, image_key=None, device=None, dtype=None, allow_prompt_reuse=True, strict=None):
        strict = self.strict if strict is None else bool(strict)
        record = self.find_record(
            prompt,
            image_key=image_key,
            allow_prompt_reuse=allow_prompt_reuse,
            validate_file=False,
        )
        if record is None:
            if strict:
                raise FileNotFoundError(f"Missing text embedding cache for image_key={normalize_image_key(image_key)}")
            return None
        try:
            state = self.load_embedding_file(record)
        except Exception:
            if strict:
                raise
            return None
        return _move_embedding_state(state, device=device or "cpu", dtype=dtype or torch.float32)

    def append_manifest_record(self, record):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        image_key = normalize_image_key(record.get("image_key"))
        if image_key:
            self.records_by_image[image_key] = record
        self.records_by_prompt[(record["prompt_hash"], record["encoder_signature"])] = record
        return record

    def register_existing_embedding(self, source_record, prompt, image_key, lq_path, hq_path):
        record = dict(source_record)
        record.update(
            {
                "image_key": normalize_image_key(image_key),
                "lq_path": normalize_image_key(lq_path),
                "hq_path": normalize_image_key(hq_path) if hq_path else None,
                "prompt_hash": compute_prompt_hash(prompt),
                "encoder_signature": self.encoder_signature,
            }
        )
        return self.append_manifest_record(record)

    def save_embedding(self, prompt, image_key, lq_path, hq_path, state):
        prompt_hash = compute_prompt_hash(prompt)
        rel_path = self.embedding_relative_path(prompt_hash)
        path = self.cache_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in state.items()
            if key in TEXT_EMBEDDING_KEYS
        }
        payload["prompt_hash"] = prompt_hash
        payload["encoder_signature"] = self.encoder_signature
        tmp_path = Path(str(path) + ".tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)

        record = {
            "image_key": normalize_image_key(image_key),
            "lq_path": normalize_image_key(lq_path),
            "hq_path": normalize_image_key(hq_path) if hq_path else None,
            "prompt_hash": prompt_hash,
            "encoder_signature": self.encoder_signature,
            "embedding_path": str(rel_path).replace("\\", "/"),
            "flux_backend": str(cfg(self.config, "model.flux_backend", "flux1") or "flux1").lower(),
            "max_prompt_sequence_length": int(cfg(self.config, "model.max_prompt_sequence_length", 128) or 0),
            "use_prompt": bool(cfg(self.config, "condition.use_prompt", True)),
            "use_suggestions": bool(cfg(self.config, "condition.use_suggestions", True)),
            "prompt_variant": cfg(self.config, "condition.prompt_variant", None),
            "include_caption": bool(cfg(self.config, "condition.include_caption", False)),
            "dtype": self.dtype_name,
        }
        return self.append_manifest_record(record)


def get_text_embedding_cache(config, dtype=None):
    mode = normalize_text_encoding_mode(config)
    cache = TextEmbeddingCache.from_config(config, dtype=dtype)
    if mode == "cached" and cache is None:
        raise ValueError("text_encoding.cache_dir is required when text_encoding.mode is cached.")
    return cache


def validate_online_prompt_lengths(artist, prompts, config):
    max_length = int(cfg(config, "model.max_prompt_sequence_length", 0) or 0)
    counter = getattr(artist, "prompt_token_lengths", None)
    if max_length <= 0 or not callable(counter):
        return
    validated = getattr(artist, "_validated_prompt_length_hashes", None)
    if validated is None:
        validated = set()
        setattr(artist, "_validated_prompt_length_hashes", validated)
    pending = [
        (index, prompt, compute_prompt_hash(prompt))
        for index, prompt in enumerate(prompts)
        if compute_prompt_hash(prompt) not in validated
    ]
    if not pending:
        return
    lengths = list(counter([prompt for _, prompt, _ in pending]))
    if len(lengths) != len(pending):
        raise RuntimeError(
            "Prompt tokenizer returned a mismatched number of token lengths."
        )
    overflows = [
        (original_index, int(length))
        for (original_index, _prompt, _prompt_hash), length in zip(
            pending,
            lengths,
        )
        if int(length) > max_length
    ]
    if overflows:
        index, length = overflows[0]
        raise ValueError(
            "Prompt exceeds model.max_prompt_sequence_length without truncation: "
            f"prompt_index={index} tokens={length} limit={max_length}"
        )
    validated.update(prompt_hash for _, _, prompt_hash in pending)


def resolve_prompt_embeddings(artist, prompts, image_keys, config, device, dtype, cache=None):
    mode = normalize_text_encoding_mode(config)
    if isinstance(prompts, str):
        prompts = [prompts]
    if image_keys is None:
        image_keys = [None] * len(prompts)
    elif isinstance(image_keys, (str, Path)):
        image_keys = [image_keys]
    image_keys = list(image_keys)
    if len(image_keys) != len(prompts):
        raise ValueError(f"image_keys length {len(image_keys)} does not match prompts length {len(prompts)}")

    if mode == "online":
        validate_online_prompt_lengths(artist, prompts, config)
        return artist.encode_prompts(prompts, device=device, dtype=dtype)

    cache = cache or get_text_embedding_cache(config, dtype=dtype)
    if cache is None:
        if mode == "cached":
            raise ValueError("text_encoding.cache_dir is required when text_encoding.mode is cached.")
        validate_online_prompt_lengths(artist, prompts, config)
        return artist.encode_prompts(prompts, device=device, dtype=dtype)

    states = []
    for prompt, image_key in zip(prompts, image_keys):
        state = cache.load_embedding(
            prompt,
            image_key=image_key,
            device="cpu",
            dtype=torch.float32,
            strict=(mode == "cached"),
        )
        if state is None:
            if mode == "cached":
                raise FileNotFoundError(f"Missing text embedding cache for image_key={normalize_image_key(image_key)}")
            validate_online_prompt_lengths(artist, prompts, config)
            return artist.encode_prompts(prompts, device=device, dtype=dtype)
        states.append(state)
    return stack_embedding_states(states, device=device, dtype=dtype)
