"""Differentiable answer-only scores and auxiliary greedy exact match."""
import re
from contextlib import contextmanager

import torch


@contextmanager
def evaluation_mode(model):
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training


def answer_inputs(model, tokenizer, prompt, answer):
    # Offset masking avoids silently treating prompt/answer boundary merges as prompt tokens.
    encoded = tokenizer(prompt + ' ' + answer, return_tensors='pt',
                        return_offsets_mapping=True, truncation=False)
    offsets = encoded.pop('offset_mapping')[0]
    if any(int(start) < len(prompt) and int(end) > len(prompt) + 1 for start, end in offsets):
        raise ValueError('A token crosses the prompt/answer boundary; use a separable prompt')
    mask = torch.tensor([int(start) >= len(prompt) and int(end) > len(prompt) + 1
                         for start, end in offsets], dtype=torch.bool)
    device = model.get_input_embeddings().weight.device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    mask = mask[1:].to(device)
    if not mask.any():
        raise ValueError('No scorable answer tokens after boundary masking')
    return encoded, mask


def mean_answer_logprob(model, tokenizer, prompt, answer):
    with evaluation_mode(model):
        encoded, mask = answer_inputs(model, tokenizer, prompt, answer)
        logits = model(**encoded).logits[:, :-1].float()
        labels = encoded['input_ids'][:, 1:]
        logprobs = logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        return logprobs[0, mask].mean()


def smooth_answer_margin(model, tokenizer, prompt, answer, competitors):
    if not competitors:
        raise ValueError('At least one competitor required')
    target = mean_answer_logprob(model, tokenizer, prompt, answer)
    other = torch.stack([mean_answer_logprob(model, tokenizer, prompt, c) for c in competitors])
    return target - torch.logsumexp(other, dim=0)


def fact_margins(model, tokenizer, fact):
    def margin(prompt):
        return smooth_answer_margin(model, tokenizer, prompt, fact['target_new'], fact['competitor_answers'])
    access = torch.stack([margin(q) for q in fact['access_questions']]).mean()
    storage = margin(fact['storage_prompt'])
    context = torch.stack([margin(fact['support_context'] + '\n' + q)
                           for q in fact['access_questions']]).mean()
    return dict(access_margin=access, storage_proxy_margin=storage,
                context_supported_margin=context, access_gap=storage - access)


def normalize_answer(text):
    return re.sub(r'\s+', ' ', text.strip()).casefold()


@torch.no_grad()
def greedy_answer(model, tokenizer, prompt, max_new_tokens):
    with evaluation_mode(model):
        inputs = tokenizer(prompt, return_tensors='pt', truncation=False)
        device = model.get_input_embeddings().weight.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def evaluate_fact(model, tokenizer, fact, latest_target, tau_storage, tau_context, max_new_tokens):
    result = {k: float(v) for k, v in fact_margins(model, tokenizer, fact).items()}
    def answers(prompts):
        return [normalize_answer(greedy_answer(model, tokenizer, p, max_new_tokens)) for p in prompts]
    target = normalize_answer(fact['target_new'])
    access = answers(fact['access_questions'])
    storage = answers([fact['storage_prompt']])
    context = answers([fact['support_context'] + '\n' + q for q in fact['access_questions']])
    result.update(access_em=sum(a == target for a in access) / len(access),
                  storage_em=float(storage[0] == target),
                  context_em=sum(a == target for a in context) / len(context),
                  latest_write_intrusion=(None if latest_target is None or normalize_answer(latest_target) == target
                                          else sum(a == normalize_answer(latest_target) for a in access) / len(access)))
    result['access_specific_failure'] = (result['access_margin'] <= 0 and
                                         result['storage_proxy_margin'] >= tau_storage and
                                         result['context_supported_margin'] >= tau_context)
    return result
