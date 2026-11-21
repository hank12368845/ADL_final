"""Completion and tool-name token spans from the official Qwen chat template."""
from __future__ import annotations
import re


def encode_target(tokenizer, messages, tools, tool, args):
    answer = {'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': 'answer', 'type': 'function', 'function': {'name': tool, 'arguments': args}}]}
    prompt = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
    full = tokenizer.apply_chat_template(messages+[answer], tools=tools, tokenize=False,
                                        add_generation_prompt=False, enable_thinking=False)
    if not full.startswith(prompt):
        raise ValueError('Assistant completion is not a continuation of the official prompt')
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    if encoded['input_ids'][:len(prefix)] != prefix:
        raise ValueError('Official template target token boundary mismatch')
    text = full[len(prompt):]
    match = re.search(r'<function=([^>]+)>', text)
    if not match or match.group(1) != tool:
        raise ValueError('Unexpected official tool syntax')
    spans = [('tool_name', match.start(1), match.end(1))]
    for match in re.finditer(r'<parameter=([^>]+)>(.*?)</parameter>', text, re.DOTALL):
        label = 'private_state' if match.group(1)=='private_state' else 'trade_arguments'
        spans.append((label, match.start(2), match.end(2)))
    regions = []
    for start, end in encoded['offset_mapping'][len(prefix):]:
        a, b = start-len(prompt), end-len(prompt)
        regions.append(next((label for label, lo, hi in spans if a < hi and b > lo), 'format'))
    positions = [i for i, label in enumerate(regions) if label=='tool_name']
    if not positions or 'private_state' not in regions:
        raise ValueError('Missing decision or memory token span')
    return {'input_ids': encoded['input_ids'], 'prompt_length': len(prefix),
            'target_length': len(regions), 'tool_name_positions': positions,
            'regions': regions, 'completion': text}
