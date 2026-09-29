"""Offline tests: randomly initialized tiny Llama, PEFT, and an editor mock."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.query_access.data import load_facts, make_stream
from experiments.query_access.interference import (
    assert_no_grad, dot_states, layer_interference, measure_gap_gradient,
)
from experiments.query_access.lora_state import (
    ensure_lora_initialized, flatten_named_gradient, flatten_named_state,
    group_state_by_layer, snapshot_lora, subtract_states,
)
from experiments.query_access.metrics import leave_stream_out, summarize
from experiments.query_access.run_probe import FILENAMES, read_rows, run_stream
from experiments.query_access.scoring import (
    answer_inputs, fact_margins, mean_answer_logprob, smooth_answer_margin,
)


torch.set_num_threads(1)


class CharTokenizer:
    pad_token_id = 0

    def __call__(self, text, return_tensors=None, return_offsets_mapping=False, **kwargs):
        ids = [1] + [ord(c) for c in text]
        result = {'input_ids': torch.tensor([ids]), 'attention_mask': torch.ones(1, len(ids), dtype=torch.long)}
        if return_offsets_mapping:
            result['offset_mapping'] = torch.tensor([[(0, 0)] + [(i, i + 1) for i in range(len(text))]])
        return result

    def decode(self, ids, skip_special_tokens=True):
        return ''.join(chr(int(i)) for i in ids if int(i) > 2)


def facts(count=5):
    return [dict(fact_id=str(i), write_prompt=f'F{i}:', target_new='a',
                 access_questions=[f'Q{i}?', f'Ask{i}?'], storage_prompt=f'F{i}:',
                 support_context=f'F{i} is a.', competitor_answers=['b', 'c']) for i in range(count)]


class MockEditor:
    def __init__(self):
        self.model = LlamaForCausalLM(LlamaConfig(
            vocab_size=128, hidden_size=8, intermediate_size=16, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128,
            bos_token_id=1, eos_token_id=2, pad_token_id=0, attention_dropout=0.0))
        self.tok = CharTokenizer()
        self.calls = []

    def edit(self, prompts, target_new, sequential_edit, verbose):
        assert len(prompts) == len(target_new) == 1
        assert sequential_edit is True and verbose is False
        self.calls.append(prompts[0])
        optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=0.03)
        optimizer.zero_grad(set_to_none=True)
        loss = -mean_answer_logprob(self.model, self.tok, prompts[0], target_new[0])
        loss.backward()
        optimizer.step()


@pytest.fixture
def setup():
    torch.manual_seed(7)
    editor = MockEditor()
    hp = SimpleNamespace(lora_type='lora', rank=2, lora_alpha=4, lora_dropout=0.0,
                         layers=[], target_modules=['q_proj', 'v_proj'], batch_size=1,
                         num_steps=1, lr=0.03)
    return editor, hp


def test_initialization_is_effectively_zero_and_first_delta_excludes_init(setup):
    editor, hp = setup
    editor.model.eval()
    inputs = editor.tok('F0: a')
    baseline = editor.model(**inputs).logits.detach().clone()
    ensure_lora_initialized(editor, hp)
    torch.testing.assert_close(editor.model(**inputs).logits, baseline, rtol=0, atol=0)
    before = snapshot_lora(editor.model)
    assert any(t.count_nonzero() for n, t in before.items() if 'lora_A' in n)
    assert all(not t.count_nonzero() for n, t in before.items() if 'lora_B' in n)
    editor.edit(['F0:'], ['a'], True, False)
    after = snapshot_lora(editor.model)
    delta = subtract_states(after, before)
    assert all(not t.count_nonzero() for n, t in delta.items() if 'lora_A' in n)
    assert any(t.count_nonzero() for n, t in delta.items() if 'lora_B' in n)
    for name in before:
        torch.testing.assert_close(before[name] + delta[name], after[name])
        assert delta[name].device.type == 'cpu' and delta[name].dtype == torch.float32


def test_scoring_masks_only_answer_and_does_not_mutate(setup):
    editor, hp = setup
    ensure_lora_initialized(editor, hp)
    before = {n: p.detach().clone() for n, p in editor.model.named_parameters()}
    encoded, mask = answer_inputs(editor.model, editor.tok, 'hello', 'ab')
    assert int(mask.sum()) == 2
    assert encoded['input_ids'][0, 1:][mask].tolist() == [ord('a'), ord('b')]
    score = mean_answer_logprob(editor.model, editor.tok, 'hello', 'ab')
    logits = editor.model(**encoded).logits[:, :-1].float().log_softmax(-1)
    expected = (logits[0, -2, ord('a')] + logits[0, -1, ord('b')]) / 2
    torch.testing.assert_close(score, expected)
    margin = smooth_answer_margin(editor.model, editor.tok, 'hello', 'ab', ['cd', 'ef'])
    assert margin.requires_grad
    for n, p in editor.model.named_parameters():
        torch.testing.assert_close(before[n], p, rtol=0, atol=0)
    assert_no_grad(editor.model)


def test_gap_and_state_interference_layer_sum_and_changing_gradient(setup):
    editor, hp = setup
    ensure_lora_initialized(editor, hp)
    fact = facts()[0]
    editor.edit([fact['write_prompt']], [fact['target_new']], True, False)
    gap, fixed = measure_gap_gradient(editor.model, editor.tok, fact)
    margins = fact_margins(editor.model, editor.tok, fact)
    torch.testing.assert_close(margins['access_gap'], margins['storage_proxy_margin'] - margins['access_margin'])
    assert gap == pytest.approx(margins['access_gap'].item(), abs=1e-6)
    before = snapshot_lora(editor.model)
    editor.edit(['unrelated?'], ['c'], True, False)
    delta = subtract_states(snapshot_lora(editor.model), before)
    full, layers = layer_interference(fixed, delta)
    flat = torch.dot(flatten_named_gradient(fixed).double(), flatten_named_state(delta).double()).item()
    assert full == pytest.approx(flat, abs=1e-9)
    assert full == pytest.approx(dot_states(fixed, delta), abs=1e-9)
    assert abs(sum(layers.values()) - full) < 1e-9
    _, current = measure_gap_gradient(editor.model, editor.tok, fact)
    assert not torch.allclose(flatten_named_gradient(fixed), flatten_named_gradient(current), atol=1e-8, rtol=1e-5)
    assert_no_grad(editor.model)


def test_state_validation_and_other_layer():
    state = {'layers.2.lora_A.weight': torch.ones(2), 'head.lora_B.weight': torch.ones(3)}
    assert set(group_state_by_layer(state)) == {'2', 'other'}
    with pytest.raises(ValueError, match='keys'):
        subtract_states(state, {})
    with pytest.raises(ValueError, match='shape'):
        subtract_states(state, {**state, 'head.lora_B.weight': torch.ones(4)})
    with pytest.raises(AssertionError):
        layer_interference(state, {n: t * float('nan') for n, t in state.items()})


def test_seed_and_validation(tmp_path):
    records = facts(20)
    assert make_stream(records, 42, 20) == make_stream(records, 42, 20)
    assert make_stream(records, 42, 20) != make_stream(records, 43, 20)
    assert {r['fact_id'] for r in make_stream(records, 42, 20)} == {r['fact_id'] for r in records}
    path = tmp_path / 'facts.jsonl'
    path.write_text('\n'.join(json.dumps(f) for f in records))
    assert load_facts(path) == records
    records[0]['access_questions'] = [records[0]['write_prompt']]
    path.write_text('\n'.join(json.dumps(f) for f in records))
    with pytest.raises(ValueError, match='held out'):
        load_facts(path)


def test_five_fact_pipeline_files_and_cumulative(setup, tmp_path):
    editor, hp = setup
    stream = make_stream(facts(), 41, 5)
    run_stream(editor, hp, stream, 41, tmp_path, max_new_tokens=1)
    trajectory, interference, layers = [read_rows(tmp_path / name) for name in FILENAMES]
    assert len(trajectory) == 5 + 15
    assert len(interference) == 10
    assert len(editor.calls) == 5
    assert len(list((tmp_path / 'seed_41/deltas').glob('*.pt'))) == 5
    manifest = json.loads((tmp_path / 'seed_41/stream_manifest.json').read_text())
    assert manifest['fact_order'] == [r['fact_id'] for r in stream]
    for row in interference:
        selected = [r['I_state_layer'] for r in layers if r['write_idx'] == row['write_idx'] and r['fact_id'] == row['fact_id']]
        assert sum(selected) == pytest.approx(row['I_state'], abs=1e-6)
    for row in trajectory:
        if row['write_idx'] >= 0:
            previous = [r for r in interference if r['fact_id'] == row['fact_id'] and r['write_idx'] <= row['write_idx']]
            assert row['cum_state'] == pytest.approx(sum(r['I_state'] for r in previous))
            assert row['cum_fixed'] == pytest.approx(sum(r['I_fixed'] for r in previous))
            assert row['cum_delta_norm'] == pytest.approx(sum(r['delta_norm'] for r in previous))
            assert row['access_specific_failure'] == (row['access_margin'] <= 0 and row['storage_proxy_margin'] >= 0 and row['context_supported_margin'] >= 0)
    summary = summarize(trajectory, interference, tmp_path, repetitions=30)
    assert summary['leave_stream_out']['status'] == 'insufficient_streams'
    assert summary['bootstrap_ci_delta_rho'] is None
    assert (tmp_path / 'query_access_summary.csv').exists()
    assert_no_grad(editor.model)


def test_leave_stream_out_has_no_leakage():
    rows = [dict(seed=s, fact_id=str(i), cum_state=i + s, cum_fixed=i - s,
                 cum_delta_norm=2 * i, access_margin=1 - i, access_gap=i,
                 access_specific_failure=i > 2) for s in (1, 2, 3) for i in range(5)]
    result = leave_stream_out(rows)
    assert result['status'] == 'ok'
    changed = copy.deepcopy(rows)
    for row in changed:
        if row['seed'] == 3:
            row['access_margin'] = 1000
            row['access_specific_failure'] = not row['access_specific_failure']
    altered = leave_stream_out(changed)
    for before, after in zip(result['predictions'], altered['predictions']):
        assert before['seed'] not in before['training_seeds']
        if before['seed'] == 3:
            assert before['prediction'] == after['prediction']
            assert before['train_threshold'] == after['train_threshold']


def test_configuration_guard(setup):
    editor, hp = setup
    hp.lora_dropout = 0.1
    with pytest.raises(ValueError, match='dropout'):
        ensure_lora_initialized(editor, hp)


def test_actual_easyedit_execute_lora_reuses_initialized_adapter(setup):
    # Load the unmodified production function without importing unrelated editor dependencies.
    import ast
    import typing
    from copy import deepcopy
    from pathlib import Path

    import peft
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    source = Path(__file__).resolve().parents[1] / 'easyeditor/models/lora/lora_main.py'
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                 and node.name in ('execute_lora', 'AverageMeter', 'chunks')]
    namespace = dict(torch=torch, deepcopy=deepcopy, normalize_device=lambda _: torch.device('cpu'),
                     AutoModelForCausalLM=object, AutoTokenizer=object, LoRAHyperParams=object,
                     **{n: getattr(typing, n) for n in ('List', 'Dict', 'Tuple', 'Any')},
                     **{n: getattr(peft, n) for n in ('LoraConfig', 'AdaLoraConfig', 'TaskType', 'get_peft_model')})
    exec(compile(tree, str(source), 'exec'), namespace)
    editor, hp = setup
    hp.model_name = 'tiny'
    hp.weight_decay = 0.0
    hp.device = 'cpu'
    tokenizer = Tokenizer(WordLevel({'[PAD]': 0, '[UNK]': 1, 'F': 3, ':': 4, 'a': 5, 'b': 6}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token='[PAD]', unk_token='[UNK]')
    ensure_lora_initialized(editor, hp)
    before = snapshot_lora(editor.model)
    result = namespace['execute_lora'](editor.model, tok, [{'prompt': 'F:', 'target_new': 'a'}], hp)
    assert result is editor.model
    after = snapshot_lora(result)
    delta = subtract_states(after, before)
    assert any(t.count_nonzero() for t in delta.values())
    # Reentrant training checkpointing is disabled before autograd.grad measurement.
    _, grad = measure_gap_gradient(result, tok, dict(storage_prompt='F:', access_questions=['b:'],
                                                    target_new='a', competitor_answers=['b']))
    assert grad.keys() == delta.keys()
    assert_no_grad(result)


def test_boundary_crossing_is_rejected(setup):
    editor, hp = setup
    ensure_lora_initialized(editor, hp)

    class MergedTokenizer(CharTokenizer):
        def __call__(self, *args, **kwargs):
            encoded = super().__call__(*args, **kwargs)
            encoded['offset_mapping'][0, -1] = torch.tensor([0, 3])
            return encoded

    with pytest.raises(ValueError, match='crosses'):
        mean_answer_logprob(editor.model, MergedTokenizer(), 'a', 'b')


def test_stream_bootstrap_and_constant_correlations():
    from experiments.query_access.metrics import association, bootstrap, correlation

    rows = [dict(seed=s, x=i + s, y=-2 * (i + s)) for s in range(3) for i in range(4)]
    result = association(rows, 'x', 'y', repetitions=30, seed=8)
    assert result['pearson'] == pytest.approx(-1)
    assert result['spearman'] == pytest.approx(-1)
    assert result['spearman_ci95'] == pytest.approx([-1, -1])
    assert correlation([1, 1, 1], [1, 2, 3]) is None

    def whole_streams(sample):
        # Every bootstrap draw includes all four facts of each sampled stream.
        for seed in range(3):
            assert sum(r['seed'] == seed for r in sample) % 4 == 0
        return 0.0

    assert bootstrap(rows, whole_streams, 30, 8) == [0.0, 0.0]
