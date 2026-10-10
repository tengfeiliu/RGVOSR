"""Strict cache reuse and F0-anchored quality decisions for the small pilot."""
from pathlib import Path
import hashlib
import random

from PIL import Image

from rl_sr.schema import StateRecord, content_sha256, identity_sha256
from rl_sr.self_refine import quality_verdict


def validate_teacher_state(state, source, f0_sha256):
    if state['round_index'] not in (2, 3, 4):
        raise ValueError('Teacher buffer has an invalid target round')
    for key, expected in [('sample_key', source['_sample_key']), ('prompt', source['_prompt']),
                          ('producer_adapter_sha256', f0_sha256)]:
        if state.get(key) != expected:
            raise ValueError(f'Teacher state {key} differs')
    expected_sampler=identity_sha256(dict(num_steps=25,schedule='linear',init_mode='pure_noise',sigma_start=1.))
    condition=source['_router_condition']
    if (state.get('sampler_sha256')!=expected_sampler or
        state.get('metadata',{}).get('router_condition')!=condition or
        state.get('router_condition_sha256')!=condition['source_hash'] or
        state.get('prompt_sha256')!=hashlib.sha256(source['_prompt'].encode()).hexdigest()):
        raise ValueError('Teacher sampler/router/prompt identity differs')
    if identity_sha256(StateRecord.from_dict(state).identity_payload())!=state['state_id']:
        raise ValueError('Teacher portable state identity differs')
    for field, source_field, hash_field in [('original_lr','lq_path','original_lr_sha256'),
                                           ('hr','hq_path','hr_sha256')]:
        if Path(state['runtime'][field]).resolve() != Path(source[source_field]).resolve():
            raise ValueError(f'Teacher state {field} is assigned to another source')
        if content_sha256(source[source_field]) != state[hash_field]:
            raise ValueError(f'Teacher state {field} content changed')
    current = Path(state['runtime']['current_sr'])
    if content_sha256(current) != state['current_output_sha256']:
        raise ValueError('Teacher state output content changed')
    with Image.open(current) as im, Image.open(source['hq_path']) as hr:
        if im.size != hr.size or im.size != (512, 512):
            raise ValueError('Teacher state geometry differs from paired 512 training')
        if state['geometry_sha256']!=identity_sha256(dict(mode=im.mode,width=im.width,height=im.height)):
            raise ValueError('Teacher geometry identity differs')
    return state


def mixed_states(old, own, fraction, seed):
    """Keep static-buffer ordering: cycle zero is exactly matched to A."""
    if not 0 < fraction <= 1:
        raise ValueError('Self-state fraction must be in (0,1]')
    key = lambda r: (r['round_index'], str(Path(r['runtime']['original_lr']).resolve()))
    a, b = {key(r):r for r in old}, {key(r):r for r in own}
    if len(a) != len(old) or len(b) != len(own) or set(a) != set(b):
        raise ValueError('Old and self buffers do not have identical source/round coverage')
    selected = set()
    for k in (2,3,4):
        keys = sorted(pair for pair in a if pair[0] == k)
        random.Random(seed+k).shuffle(keys)
        selected.update(keys[:round(len(keys)*fraction)])
    return [b[key(r)] if key(r) in selected else r for r in old]


def passes_first_round_gate(first, late, raw_early, thresholds):
    return (quality_verdict(first, late, first, thresholds)['scale_ok'] and
            quality_verdict(raw_early, late, raw_early, thresholds)['scale_ok'])
