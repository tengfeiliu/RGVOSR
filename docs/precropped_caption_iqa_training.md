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

Generate crop-invariant suggestions from the four IQA fields only:

```bash
python tools/generate_iqa_sr_suggestion_jsonl.py \
  --input datasets/LSDIR_precrop512/train.iqa_caption.jsonl \
  --output datasets/LSDIR_precrop512/train.iqa_caption_suggestion.jsonl \
  --resume
```

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
