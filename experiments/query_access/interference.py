"""First-order gap interference, measured without accumulating .grad."""
import math

import torch

from .lora_state import check_states, group_state_by_layer, lora_parameters
from .scoring import smooth_answer_margin


def prepare_measurement(model):
    model.zero_grad(set_to_none=True)
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    model.eval()


def assert_no_grad(model):
    if any(p.grad is not None for p in model.parameters()):
        raise AssertionError('Gradient measurement left .grad buffers')


def measure_gap_gradient(model, tokenizer, fact):
    prepare_measurement(model)
    params = lora_parameters(model)
    result = {n: torch.zeros_like(p, device='cpu', dtype=torch.float32) for n, p in params.items()}
    terms = [(fact['storage_prompt'], 1.0)] + [
        (q, -1.0 / len(fact['access_questions'])) for q in fact['access_questions']]
    gap = 0.0
    for prompt, weight in terms:
        score = smooth_answer_margin(model, tokenizer, prompt, fact['target_new'], fact['competitor_answers'])
        gap += weight * score.detach().item()
        grads = torch.autograd.grad(score, tuple(params.values()), allow_unused=True)
        for (name, param), grad in zip(params.items(), grads):
            if grad is not None:
                result[name].add_(grad.detach().cpu().float(), alpha=weight)
    assert_no_grad(model)
    return gap, result


def dot_states(grads, delta):
    check_states(grads, delta)
    return sum((grads[n].double() * delta[n].double()).sum().item() for n in grads)


def layer_interference(grads, delta, tolerance=1e-6):
    check_states(grads, delta)
    gg, dd = group_state_by_layer(grads), group_state_by_layer(delta)
    layers = {layer: dot_states(gg[layer], dd[layer]) for layer in gg}
    full = dot_states(grads, delta)
    if not math.isfinite(full) or abs(sum(layers.values()) - full) >= tolerance:
        raise AssertionError('Layer decomposition does not sum to full interference')
    return full, layers
