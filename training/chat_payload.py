"""Boundary between HF template dictionaries and Chat Completions JSON strings."""
from __future__ import annotations
import copy
import json


def api_messages(messages):
    """Serialize outer function arguments only; private_state remains a JSON string.

    HF templates want dict arguments; the HTTP API requires string arguments and
    vLLM decodes them before applying the same template. Never mutate frozen data.
    """
    result = copy.deepcopy(messages)
    for message in result:
        for call in message.get('tool_calls', []):
            function = call['function']
            value = function.get('arguments', {})
            if isinstance(value, str):
                parsed = json.loads(value) if value else {}
                if not isinstance(parsed, dict):
                    raise ValueError('Historical tool arguments must encode an object')
                function['arguments'] = value or '{}'
            elif isinstance(value, dict):
                function['arguments'] = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
            else:
                raise ValueError('Historical tool arguments must be an object or JSON object string')
    return result


def validate_payloads(cases, personas):
    """Use the installed grading-version vLLM request schema, entirely on CPU."""
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from training.buyer_diagnostics import replace_persona
    checked = 0
    for case in cases:
        for context in case['contexts']:
            for persona in personas.values():
                messages = api_messages(replace_persona(case['messages'], persona, context))
                ChatCompletionRequest.model_validate({
                    'model': 'Qwen/Qwen3.5-4B', 'messages': messages, 'tools': case['tools'],
                    'tool_choice': 'auto', 'max_tokens': min(8192,case['observation'].get('output_tokens_left_this_round',8192)),
                    'temperature': .7, 'top_p': .95, 'chat_template_kwargs': {'enable_thinking': False}})
                checked += 1
    return {'wire_format': 'openai_tool_arguments_string_v1', 'validated_requests': checked,
            'validator': 'vllm.ChatCompletionRequest', 'gpu_work_started': False}
