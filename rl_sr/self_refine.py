"""Auditable data identities, resource admission and pilot decisions."""
from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image

from rl_sr.schema import RuntimePaths, StateRecord, content_sha256, identity_sha256


def source_group(row):
    """Use original HR identity, before the pre-crop expansion."""
    value = row.get("source_hq_path") or row.get("hq_source_path") or row.get("hq_path")
    if not value:
        raise ValueError("Source lacks HR identity")
    return str(Path(value).expanduser().resolve())


def split_sources(rows, baseline_lq_paths, train_count, val_count, test_count, seed):
    baseline = {str(Path(p).expanduser().resolve()) for p in baseline_lq_paths}
    training_groups = {source_group(r) for r in rows
                       if str(Path(r['lq_path']).expanduser().resolve()) in baseline}
    train, holdout = {}, {}
    for row in rows:
        group = source_group(row)
        lr = str(Path(row['lq_path']).expanduser().resolve())
        if lr in baseline:
            train.setdefault(group, row)
        elif group not in training_groups:
            holdout.setdefault(group, row)
    rng = random.Random(seed)
    a, b = sorted(train), sorted(holdout)
    rng.shuffle(a); rng.shuffle(b)
    chosen = [train[g] for g in a[:train_count]]
    # Full baseline scale has 3000 crops from 2895 HR sources. Multiple training
    # crops are allowed; the independent heldout exclusion still uses every HR group.
    selected_lr = {r['lq_path'] for r in chosen}
    extra = [r for r in rows if str(Path(r['lq_path']).expanduser().resolve()) in baseline
             and r['lq_path'] not in selected_lr]
    if len(chosen)<train_count:
        chosen.extend(extra[:train_count-len(chosen)])
    if len(chosen) < train_count or len(b) < val_count + test_count:
        raise ValueError(f"Insufficient independent HR groups: train={len(a)}, heldout={len(b)}")
    return dict(train=chosen,
                validation=[holdout[g] for g in b[:val_count]],
                test=[holdout[g] for g in b[val_count:val_count+test_count]])


def states_from_trajectory(sources, trajectory, old_hash, f0_hash, sampler_hash):
    index = {}
    for row in trajectory:
        key = (row['sample_key'], int(row['round']))
        if key in index:
            raise ValueError(f"Duplicate trajectory output {key}")
        index[key] = row
    records = []
    for source in sources:
        key = source['_sample_key']
        if not all((key, k) in index for k in range(1, 5)):
            raise ValueError(f"Incomplete 4-round trajectory: {key}")
        lr, hr = source['lq_path'], source['hq_path']
        lr_hash, hr_hash = content_sha256(lr), content_sha256(hr)
        parent = None
        for target in range(2, 5):
            current = index[(key, target-1)]['path']
            child = index[(key,target)]
            if child.get('current_sr') and Path(child['current_sr']).resolve()!=Path(current).resolve():
                raise ValueError(f'Trajectory has wrong current parent at round {target}')
            if child.get('parent_sha256') and child['parent_sha256']!=content_sha256(current):
                raise ValueError(f'Trajectory parent hash mismatch at round {target}')
            with Image.open(current) as im:
                geometry = identity_sha256(dict(mode=im.mode, width=im.width, height=im.height))
            record = StateRecord(
                dataset_id='lsdir_precrop512_v1', sample_key=key,
                original_lr_key=f'original_lr/sha256/{lr_hash}', original_lr_sha256=lr_hash,
                current_output_key=f'current/sha256/{content_sha256(current)}',
                current_output_sha256=content_sha256(current), round_index=target,
                parent_state_id=parent, geometry_sha256=geometry,
                producer_adapter_sha256=f0_hash if target == 2 else old_hash,
                sampler_sha256=sampler_hash, hr_key=f'hr/sha256/{hr_hash}', hr_sha256=hr_hash,
                prompt=source.get('_prompt', ''),
                metadata={'router_condition': source.get('_router_condition'),
                          'original_hr_group': source_group(source)},
                runtime=RuntimePaths(original_lr=lr, current_sr=current, hr=hr))
            parent = record.state_id
            records.append(record)
    return records


def admit_stage(stage, baseline_phase, checkpoint_ready, free_gib, baseline_active,
                required_gib=None):
    if stage in {'prepare', 'verdict', 'metrics', 'report'}:
        return True, 'CPU stage'
    if not checkpoint_ready:
        return False, 'Waiting for complete baseline C checkpoint'
    if baseline_active and baseline_phase not in {'04', '06'}:
        return False, f'Baseline phase {baseline_phase} is not a safe concurrent window'
    if free_gib is None or not math.isfinite(free_gib):
        return False, 'CUDA memory probe unavailable'
    required = required_gib or (40.0 if stage.startswith('train_') and
                              'controller' not in stage else 24.0)
    if free_gib < required:
        return False, f'Free {free_gib:.1f} GiB < admission threshold {required:.1f} GiB'
    return True, f'CUDA free {free_gib:.1f} GiB; baseline phase {baseline_phase}'


def quality_verdict(base, candidate, early, thresholds=None):
    t = dict(psnr_tolerance=.05, ssim_tolerance=.002,
             lpips_gain=.002, dists_gain=.001, recursive_lpips_gain=.001,
             recursive_dists_gain=.001)
    t.update(thresholds or {})
    for row in (base, candidate, early):
        if not all(math.isfinite(float(row[k])) for k in ('psnr', 'ssim', 'lpips', 'dists')):
            raise ValueError('Nonfinite paired quality metric')
    fidelity = (candidate['psnr'] >= base['psnr']-t['psnr_tolerance'] and
                candidate['ssim'] >= base['ssim']-t['ssim_tolerance'])
    improves = (fidelity and candidate['lpips'] <= base['lpips']-t['lpips_gain'] and
                candidate['dists'] <= base['dists']-t['dists_gain'])
    recursive = (candidate['psnr'] >= early['psnr']-t['psnr_tolerance'] and
                 candidate['ssim'] >= early['ssim']-t['ssim_tolerance'] and
                 candidate['lpips'] <= early['lpips']-t['recursive_lpips_gain'] and
                 candidate['dists'] <= early['dists']-t['recursive_dists_gain'])
    return dict(fidelity_ok=fidelity, improves_baseline=improves, recursive_gain=recursive,
                scale_ok=improves and recursive, thresholds=t)
