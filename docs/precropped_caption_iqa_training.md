# Offline 512×512 crop cache

The crop-local training workflow first creates two deterministic HQ crops per
source image, degrades each crop independently, and runs UniPercept on the
saved LQ crop for one caption plus four IQA fields:

```bash
python tools/generate_precropped_unipercept_cache.py \
  --input configs/train_txt/train_dataset_txt.txt \
  --hq-output-dir datasets/LSDIR_precrop512/hq \
  --lq-output-dir datasets/LSDIR_precrop512/lq \
  --output datasets/LSDIR_precrop512/train.iqa_caption.jsonl \
  --invalid-output datasets/LSDIR_precrop512/invalid.jsonl \
  --crop-size 512 \
  --crops-per-image 2 \
  --profile-sections caption iqa \
  --no-reward-scores \
  --resume
```

Generate crop-local, location-aware suggestions from the four IQA fields only.
The LLM selects degradation types and quotes crop-local location evidence; the
tool validates that evidence and compiles the final suggestion deterministically.
Suggestions are bounded to 100 English words:

```bash
python tools/generate_iqa_sr_suggestion_jsonl.py \
  --input datasets/LSDIR_precrop512/train.iqa_caption.jsonl \
  --output datasets/LSDIR_precrop512/train.iqa_caption_suggestion.jsonl \
  --resume
```

Location output is enabled by default for the pre-cropped workflow. Legacy
full-image datasets that still apply a random crop during training must pass
`--no-include-location` so their suggestions remain crop-invariant:

```bash
python tools/generate_iqa_sr_suggestion_jsonl.py \
  --input datasets/legacy/train.iqa.jsonl \
  --output datasets/legacy/train.iqa_suggestion.jsonl \
  --no-include-location \
  --resume
```

Existing completed suggestion JSONL records are not rewritten by `--resume`.
Generate a new output JSONL when migrating from location-free suggestions.
Prompt text changes invalidate old text embedding caches.

Train either backend with the same JSONL:

```bash
python train_rg_flux_sr.py \
  --config configs/train_rg_flux2_klein_sr_stage0b_512_prompt_curriculum_precropped.yaml

python train_rg_flux_sr.py \
  --config configs/train_rg_flux2_klein_sr_moe_stage0b_512_prompt_curriculum_precropped.yaml
```

`data.pre_cropped` defaults to `true`. In this mode both HQ and LQ must already
be RGB 512×512; the Dataset performs no resize, crop, or padding. Legacy
full-image configs must set `data.pre_cropped: false`.
