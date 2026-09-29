"""Run with python -m experiments.query_access.run_probe (see README)."""
import argparse
import gc
import importlib.metadata
import json
import math
import random
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .data import load_facts, make_stream, save_manifest
from .interference import assert_no_grad, dot_states, layer_interference, measure_gap_gradient, prepare_measurement
from .lora_state import ensure_lora_initialized, group_state_by_layer, snapshot_lora, state_norm, subtract_states
from .metrics import summarize
from .scoring import evaluate_fact, greedy_answer, normalize_answer


FILENAMES = ('query_access_trajectory.jsonl', 'query_access_interference.jsonl',
             'query_access_layer_interference.jsonl')


def append(path, row):
    with Path(path).open('a') as file:
        file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def run_stream(editor, hparams, stream, seed, output, tau_storage=0.0, tau_context=0.0,
               max_new_tokens=32, provenance=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    folder = output / f'seed_{seed}'
    folder.mkdir(exist_ok=False)
    for name in ('deltas', 'fixed_gradients', 'state_gradients'):
        (folder / name).mkdir()
    for name in FILENAMES:
        (output / name).touch(exist_ok=True)
    ensure_lora_initialized(editor, hparams)
    model, tokenizer = editor.model, editor.tok
    prepare_measurement(model)
    save_manifest(folder / 'stream_manifest.json', stream, seed, {
        'hparams': vars(hparams), 'tau_storage': tau_storage, 'tau_context': tau_context,
        'max_new_tokens': max_new_tokens, 'provenance': provenance,
        'context_join': 'support_context + newline + question',
        'score_join': 'prompt + space + answer',
        'cumulative_start': 'later writes only, strictly after own write',
    })
    cumulative = [dict(cum_state=0.0, cum_fixed=0.0, cum_delta_norm=0.0) for _ in stream]
    def evaluate(idx, write_idx):
        row = dict(seed=seed, write_idx=write_idx, eval_fact_idx=idx,
                   fact_id=stream[idx]['fact_id'], age=None if write_idx < 0 else write_idx - idx)
        row.update(evaluate_fact(editor.model, tokenizer, stream[idx],
                                 None if write_idx < 0 else stream[write_idx]['target_new'],
                                 tau_storage, tau_context, max_new_tokens))
        row.update(cumulative[idx])
        if idx == write_idx:
            response = greedy_answer(editor.model, tokenizer, stream[idx]['write_prompt'], max_new_tokens)
            row['write_em'] = float(normalize_answer(response) == normalize_answer(stream[idx]['target_new']))
        else:
            row['write_em'] = None
        return row
    for idx in range(len(stream)):
        append(output / FILENAMES[0], evaluate(idx, -1))
    for write_idx, fact in enumerate(stream):
        before_gap = {}
        for idx in range(write_idx):
            gap, grad = measure_gap_gradient(editor.model, tokenizer, stream[idx])
            before_gap[idx] = gap
            torch.save(grad, folder / 'state_gradients' / f'fact_{idx:04d}.pt')
            del grad
        before = snapshot_lora(editor.model)
        editor.model.train()
        editor.edit(prompts=[fact['write_prompt']], target_new=[fact['target_new']],
                    sequential_edit=True, verbose=False)
        prepare_measurement(editor.model)
        after = snapshot_lora(editor.model)
        delta = subtract_states(after, before)
        del before, after
        norm = state_norm(delta)
        metadata = dict(seed=seed, fact_id=fact['fact_id'], write_idx=write_idx,
                        delta_norm=norm, per_layer_delta_norm={k: state_norm(v) for k, v in group_state_by_layer(delta).items()},
                        num_steps=hparams.num_steps, lr=hparams.lr, rank=hparams.rank)
        torch.save({'delta': delta, **metadata}, folder / 'deltas' / f'write_{write_idx:04d}.pt')
        for idx in range(write_idx):
            state_path = folder / 'state_gradients' / f'fact_{idx:04d}.pt'
            grad = torch.load(state_path, map_location='cpu', weights_only=True)
            state_i, layers = layer_interference(grad, delta)
            del grad
            fixed = torch.load(folder / 'fixed_gradients' / f'fact_{idx:04d}.pt', map_location='cpu', weights_only=True)
            fixed_i = dot_states(fixed, delta)
            del fixed
            cumulative[idx]['cum_state'] += state_i
            cumulative[idx]['cum_fixed'] += fixed_i
            cumulative[idx]['cum_delta_norm'] += norm
            row = evaluate(idx, write_idx)
            key = dict(seed=seed, write_idx=write_idx, eval_fact_idx=idx, fact_id=stream[idx]['fact_id'], age=write_idx - idx)
            append(output / FILENAMES[1], {**key, 'actual_delta_D': row['access_gap'] - before_gap[idx],
                                          'D_before': before_gap[idx], 'D_after': row['access_gap'],
                                          'I_state': state_i, 'I_fixed': fixed_i, 'delta_norm': norm})
            for layer, value in layers.items():
                append(output / FILENAMES[2], {**key, 'layer': layer, 'I_state_layer': value})
            append(output / FILENAMES[0], row)
            state_path.unlink()
        append(output / FILENAMES[0], evaluate(write_idx, write_idx))
        _, fixed = measure_gap_gradient(editor.model, tokenizer, fact)
        torch.save(fixed, folder / 'fixed_gradients' / f'fact_{write_idx:04d}.pt')
        del fixed, delta
        assert_no_grad(editor.model)
    (folder / 'state_gradients').rmdir()
    (folder / 'completed.json').write_text(json.dumps({'seed': seed, 'num_facts': len(stream)}) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--hparams', default='hparams/LoRA/qwen2.5-7b.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', choices=['smoke', 'gate-1'], default='smoke')
    parser.add_argument('--seeds', type=int, nargs='+', default=[41])
    parser.add_argument('--tau-storage', type=float, default=0.0)
    parser.add_argument('--tau-context', type=float, default=0.0)
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--bootstrap-repetitions', type=int, default=1000)
    parser.add_argument('--model-path', help='Optional local Qwen2.5 model path')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--smoke-output', help='Completed real-model smoke output, required for Gate-1')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be unique')
    if args.max_new_tokens < 1 or args.bootstrap_repetitions < 20:
        parser.error('Require max-new-tokens >= 1 and bootstrap-repetitions >= 20')
    if not math.isfinite(args.tau_storage) or not math.isfinite(args.tau_context):
        parser.error('Storage/context thresholds must be finite')
    if args.stage == 'smoke' and len(args.seeds) != 1:
        parser.error('Smoke requires exactly 1 seed')
    if args.stage == 'gate-1':
        if len(args.seeds) < 3:
            parser.error('Gate-1 requires at least 3 independent stream seeds')
        if not args.smoke_output:
            parser.error('Run smoke first and supply --smoke-output')
        completed = Path(args.smoke_output) / 'run_completed.json'
        if not completed.exists() or json.loads(completed.read_text()).get('stage') != 'smoke':
            parser.error('A completed real-model smoke run is required')
    facts = load_facts(args.data)
    count = 5 if args.stage == 'smoke' else 20
    make_stream(facts, args.seeds[0], count)
    if Path(args.output).exists() and any(Path(args.output).iterdir()):
        parser.error('Output directory must be new or empty; partial runs are not resumed')
    # Lazy import: unit tests do not need EasyEdit's optional editor dependencies.
    from easyeditor import BaseEditor, LoRAHyperParams
    if not torch.cuda.is_available():
        raise RuntimeError('Real Qwen probe requires a CUDA runtime; use tiny tests on CPU')
    hparams = LoRAHyperParams.from_hparams(args.hparams)
    hparams.lora_dropout = 0.0
    hparams.batch_size = 1
    hparams.device = args.device
    if args.model_path:
        hparams.model_name = args.model_path
    provenance = {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  'versions': {package: importlib.metadata.version(package)
                               for package in ('torch', 'transformers', 'peft')},
                  'source_hparams': str(Path(args.hparams).resolve()), 'stage': args.stage}
    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        editor = BaseEditor.from_hparams(hparams)
        run_stream(editor, hparams, make_stream(facts, seed, count), seed, args.output,
                   args.tau_storage, args.tau_context, args.max_new_tokens, provenance)
        del editor
        gc.collect()
        torch.cuda.empty_cache()
    output = Path(args.output)
    summarize(read_rows(output / FILENAMES[0]), read_rows(output / FILENAMES[1]), output,
              args.bootstrap_repetitions)
    (output / 'run_completed.json').write_text(json.dumps({
        'stage': args.stage, 'seeds': args.seeds, 'num_facts': count,
        'hparams': asdict(hparams), 'tau_storage': args.tau_storage,
        'tau_context': args.tau_context, 'provenance': provenance}, indent=2) + '\n')


if __name__ == '__main__':
    main()
