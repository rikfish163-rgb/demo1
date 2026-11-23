"""Keep the complete pi0.5 task and discretized state in stream prompts."""
from __future__ import annotations

from threading import RLock

MAX_PROMPT_TOKENS = 512
PROMPT_INTEGRITY = {
    'schema': 'fr5-stream-prompt-integrity-v1',
    'max_tokens': MAX_PROMPT_TOKENS,
    'short_prompt_padding': 'saved_tokenizer_max_length',
    'long_prompt_padding': MAX_PROMPT_TOKENS,
}
_LOCK = RLock()


def _registry_name(step):
    return getattr(type(step), '_registry_name', None)


def _prompt_steps(preprocessor):
    steps = getattr(preprocessor, 'steps', ())
    state = [i for i, step in enumerate(steps)
             if _registry_name(step) == 'pi05_prepare_state_tokenizer_processor_step']
    tokenizer = [i for i, step in enumerate(steps) if _registry_name(step) == 'tokenizer_processor']
    if len(state) != 1 or len(tokenizer) != 1 or tokenizer[0] != state[0] + 1:
        raise ValueError('Expected the pi0.5 state prompt step immediately before one tokenizer')
    if not isinstance(getattr(preprocessor, 'before_step_hooks', None), list) or not isinstance(
            getattr(preprocessor, 'after_step_hooks', None), list):
        raise ValueError('Prompt integrity requires processor step hooks')
    return tokenizer[0], steps[state[0]], steps[tokenizer[0]]


def _check_state_prompts(transition, state_step, prompts):
    import numpy as np

    observation = transition.get('observation')
    state = observation.get('observation.state') if isinstance(observation, dict) else None
    if state is None or not hasattr(state, 'detach'):
        raise ValueError('Prompt integrity requires the normalized state tensor')
    values = state.detach().cpu().numpy()
    expected_dim = getattr(state_step, 'max_state_dim', 32)
    if (values.ndim != 2 or values.shape != (len(prompts), expected_dim)
            or not np.isfinite(values).all()):
        raise ValueError('Prompt integrity requires the complete batched pi0.5 state')
    bins = np.digitize(values, bins=np.linspace(-1, 1, 257)[:-1]) - 1
    for prompt, state_bins in zip(prompts, bins, strict=True):
        suffix = ', State: ' + ' '.join(map(str, state_bins)) + ';\nAction: '
        if not isinstance(prompt, str) or not prompt.endswith(suffix):
            raise ValueError('Prompt is missing the complete state or action marker')


def _untruncated_ids(tokenizer, texts):
    encoded = tokenizer(texts, truncation=False, padding=False, add_special_tokens=True)
    ids = encoded.get('input_ids')
    if not isinstance(ids, (list, tuple)) or len(ids) != len(texts):
        raise ValueError('Tokenizer did not return one complete token sequence per prompt')
    if any(not isinstance(sequence, (list, tuple)) for sequence in ids):
        raise ValueError('Tokenizer returned an invalid token sequence')
    return [list(sequence) for sequence in ids]


def _check_tokenized(transition, expected, token_key, mask_key):
    observation = transition.get('observation')
    if not isinstance(observation, dict):
        raise ValueError('Tokenizer did not return an observation')
    tokens, masks = observation.get(token_key), observation.get(mask_key)
    if tokens is None or masks is None or tuple(tokens.shape) != tuple(masks.shape):
        raise ValueError('Tokenizer did not return matching token and attention masks')
    if len(tokens.shape) != 2 or tokens.shape[0] != len(expected):
        raise ValueError('Tokenizer returned an invalid batched token shape')
    for row, mask, full in zip(tokens, masks, expected, strict=True):
        active = row[mask.bool()].detach().cpu().tolist()
        if active != full:
            raise ValueError('Tokenizer output does not preserve the complete prompt')


def preprocess(preprocessor, batch):
    """Use the official processor, expanding only batches whose prompts need it.

    The temporary hooks see the actual normalized state and complete prompt just
    before tokenization.  Both the hooks and the saved max length are restored
    on every exit, including errors; normal processor serialization is unchanged.
    """
    with _LOCK:
        index, state_step, tokenizer_step = _prompt_steps(preprocessor)
        original_length = getattr(tokenizer_step, 'max_length', None)
        if type(original_length) is not int or not 1 <= original_length <= MAX_PROMPT_TOKENS:
            raise ValueError(f'Saved tokenizer max_length must be between 1 and {MAX_PROMPT_TOKENS}')
        observed = {}

        def before(step_index, transition):
            if step_index != index:
                return
            prompts = tokenizer_step.get_task(transition)
            if not isinstance(prompts, list) or not prompts:
                raise ValueError('Prompt integrity requires a nonempty task batch')
            _check_state_prompts(transition, state_step, prompts)
            task_ids = _untruncated_ids(tokenizer_step.input_tokenizer, prompts)
            subtask = tokenizer_step.get_subtask(transition)
            subtask_ids = _untruncated_ids(tokenizer_step.input_tokenizer, subtask) if subtask else None
            all_ids = task_ids + (subtask_ids or [])
            longest = max(map(len, all_ids))
            if longest > MAX_PROMPT_TOKENS:
                raise ValueError(f'Complete pi0.5 prompt has {longest} tokens; maximum is {MAX_PROMPT_TOKENS}')
            tokenizer_step.max_length = original_length if longest <= original_length else MAX_PROMPT_TOKENS
            observed.update(task_ids=task_ids, subtask_ids=subtask_ids)

        def after(step_index, transition):
            if step_index != index:
                return
            if 'task_ids' not in observed:
                raise ValueError('Tokenizer ran without the prompt integrity check')
            _check_tokenized(transition, observed['task_ids'],
                             'observation.language.tokens', 'observation.language.attention_mask')
            if observed['subtask_ids'] is not None:
                _check_tokenized(transition, observed['subtask_ids'],
                                 'observation.subtask.tokens', 'observation.subtask.attention_mask')
            observed['verified'] = True

        preprocessor.before_step_hooks.append(before)
        preprocessor.after_step_hooks.append(after)
        try:
            output = preprocessor(batch)
            if not observed.get('verified'):
                raise ValueError('Processor did not run the prompt integrity check')
            return output
        finally:
            tokenizer_step.max_length = original_length
            preprocessor.before_step_hooks.remove(before)
            preprocessor.after_step_hooks.remove(after)
