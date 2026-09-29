"""Named, CPU FP32 LoRA states; no changes to EasyEdit algorithms."""
import re

import torch


def lora_parameters(model):
    params = {n: p for n, p in model.named_parameters() if p.requires_grad and 'lora_' in n}
    if not params:
        raise ValueError('No trainable LoRA parameters')
    if any(p.requires_grad and 'lora_' not in n for n, p in model.named_parameters()):
        raise ValueError('Only LoRA parameters may be trainable')
    return params


def ensure_lora_initialized(editor, hparams):
    if hparams.lora_dropout != 0.0 or hparams.batch_size != 1:
        raise ValueError('Probe requires lora_dropout=0.0 and batch_size=1')
    if hparams.lora_type != 'lora':
        raise ValueError('This probe supports standard LoRA only')
    if not hasattr(editor.model, 'peft_config'):
        from peft import LoraConfig, TaskType, get_peft_model
        editor.model = get_peft_model(editor.model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=hparams.rank, lora_alpha=hparams.lora_alpha,
            lora_dropout=hparams.lora_dropout,
            layers_to_transform=hparams.layers or None,
            target_modules=hparams.target_modules,
        ))
    else:
        configs = list(editor.model.peft_config.values())
        if len(configs) != 1:
            raise ValueError('Expected exactly one adapter')
        cfg = configs[0]
        if str(cfg.task_type) not in ('CAUSAL_LM', 'TaskType.CAUSAL_LM') or cfg.inference_mode:
            raise ValueError('Expected a trainable CAUSAL_LM adapter')
        for name, expected in [('r', hparams.rank), ('lora_alpha', hparams.lora_alpha),
                               ('lora_dropout', 0.0), ('layers_to_transform', hparams.layers or None)]:
            if getattr(cfg, name) != expected:
                raise ValueError(f'Existing adapter mismatch: {name}')
        if set(cfg.target_modules) != set(hparams.target_modules):
            raise ValueError('Existing adapter target_modules mismatch')
    assert hasattr(editor.model, 'peft_config')
    lora_parameters(editor.model)


def snapshot_lora(model):
    return {n: p.detach().to(device='cpu', dtype=torch.float32).clone()
            for n, p in lora_parameters(model).items()}


def check_states(a, b):
    if a.keys() != b.keys():
        raise ValueError('LoRA state keys differ')
    for n in a:
        if a[n].shape != b[n].shape:
            raise ValueError(f'LoRA shape mismatch: {n}')


def subtract_states(after, before):
    check_states(after, before)
    return {n: after[n].detach().cpu().float() - before[n].detach().cpu().float() for n in after}


def flatten_named_state(state):
    return torch.cat([state[n].detach().cpu().float().reshape(-1) for n in sorted(state)])


def flatten_named_gradient(grads):
    return flatten_named_state(grads)


def group_state_by_layer(state):
    groups = {}
    for name, tensor in state.items():
        match = re.search(r'(?:^|\.)layers\.(\d+)(?:\.|$)', name)
        layer = match.group(1) if match else 'other'
        groups.setdefault(layer, {})[name] = tensor
    return groups


def state_norm(state):
    return sum(t.double().square().sum().item() for t in state.values()) ** 0.5
