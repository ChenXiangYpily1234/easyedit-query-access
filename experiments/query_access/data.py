"""Validated fact records and reproducible stream manifests."""
import hashlib
import json
import random
from pathlib import Path


def load_facts(path):
    facts = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = set()
    for fact in facts:
        for key in ('fact_id', 'write_prompt', 'target_new', 'storage_prompt', 'support_context'):
            if not isinstance(fact.get(key), str) or not fact[key].strip():
                raise ValueError(f'Nonempty string required: {key}')
        for key in ('access_questions', 'competitor_answers'):
            if not isinstance(fact.get(key), list) or not fact[key] or any(
                    not isinstance(x, str) or not x.strip() for x in fact[key]):
                raise ValueError(f'Nonempty string list required: {key}')
            if len(set(fact[key])) != len(fact[key]):
                raise ValueError(f'Duplicate entries: {key}')
        if any(q.strip() in (fact['write_prompt'].strip(), fact['storage_prompt'].strip())
               for q in fact['access_questions']):
            raise ValueError('Access questions must be held out from write/storage prompts')
        if fact['target_new'].strip().casefold() in {x.strip().casefold() for x in fact['competitor_answers']}:
            raise ValueError('Target cannot also be a competitor')
        if fact['fact_id'] in ids:
            raise ValueError('Duplicate fact_id')
        if '{}' in fact['write_prompt']:
            raise ValueError('Use a resolved write_prompt, without subject placeholders')
        ids.add(fact['fact_id'])
    if not facts:
        raise ValueError('Empty fact dataset')
    return facts


def make_stream(facts, seed, count):
    if not 1 <= count <= min(len(facts), 20):
        raise ValueError('R0–R3 supports 1–20 facts; supply enough records')
    # Keep the same cohort across seeds; only its order changes.
    stream = list(facts[:count])
    random.Random(seed).shuffle(stream)
    return stream


def save_manifest(path, stream, seed, config):
    serialized = json.dumps(stream, ensure_ascii=False, sort_keys=True)
    manifest = {'seed': seed, 'fact_order': [f['fact_id'] for f in stream],
                'facts': stream, 'data_sha256': hashlib.sha256(serialized.encode()).hexdigest(),
                'config': config}
    Path(path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest
