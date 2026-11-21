"""CPU tests at the actual vLLM schema / OpenAI HTTP boundary, without a GPU."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import nullcontext

from training.buyer_operations_data import make_record, normalized, personas
from training.chat_payload import api_messages, validate_payloads


class ChatPayloadTests(unittest.TestCase):
    def case(self):
        record=make_record(29,'failed_pair',0)
        return {**record,'name':record['id'],'contexts':['recorded'],'state_mode':'operations'}

    def test_original_bug_rejected_and_wire_format_preserves_training_history(self):
        from pydantic import ValidationError
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        from vllm.entrypoints.chat_utils import _postprocess_messages
        case=self.case();original=copy.deepcopy(case['messages'])
        with self.assertRaises(ValidationError):
            ChatCompletionRequest.model_validate({'model':'Qwen/Qwen3.5-4B','messages':copy.deepcopy(original)})
        wire=api_messages(original)
        ChatCompletionRequest.model_validate({'model':'Qwen/Qwen3.5-4B','messages':copy.deepcopy(wire),'tools':case['tools']})
        self.assertTrue(all(isinstance(c['function']['arguments'],str) for m in wire for c in m.get('tool_calls',[])))
        _postprocess_messages(wire)
        self.assertEqual(wire,normalized(original))
        self.assertEqual(case['messages'],original)
        self.assertEqual(validate_payloads([case],{'p1':personas()[0]})['validated_requests'],1)

    def test_real_model_client_sends_schema_valid_json_and_parses_native_response(self):
        import httpx
        from openai import OpenAI
        from adlmarket.client import ModelClient
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        case=self.case();received=[]
        def handler(request):
            payload=json.loads(request.content)
            ChatCompletionRequest.model_validate(payload)
            received.append(payload)
            return httpx.Response(200,json={
                'id':'chatcmpl-cpu','object':'chat.completion','created':0,'model':payload['model'],
                'choices':[{'index':0,'message':{'role':'assistant','content':None,'tool_calls':[
                    {'id':'call_result','type':'function','function':{'name':'wait','arguments':json.dumps({
                        'private_state':json.dumps({'bad':[],'pending':None,'last_buy':None})})}}]},
                    'finish_reason':'tool_calls'}],
                'usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120}})
        http_client=httpx.Client(transport=httpx.MockTransport(handler))
        native=OpenAI(api_key='EMPTY',base_url='http://cpu.invalid/v1',http_client=http_client)
        client=ModelClient(api_key='EMPTY',base_url='http://cpu.invalid/v1',retry_limit=0,min_interval=0)
        client._openai_clients['http://cpu.invalid/v1\x00EMPTY']=native
        result=client.complete(model='Qwen/Qwen3.5-4B',messages=api_messages(case['messages']),
                               tools=case['tools'],max_tokens=8192,extra_body={'chat_template_kwargs':{'enable_thinking':False}})
        self.assertEqual(result.outcome,'accepted');self.assertEqual(result.tool_call.name,'wait')
        self.assertIsInstance(result.tool_call.arguments['private_state'],str)
        self.assertEqual(len(received),1)
        native.close()

    def test_infrastructure_failure_marks_run_invalid_and_never_writes_scores(self):
        from adlmarket.setups import market_run
        from training.buyer_diagnostics import worker
        root=Path(__file__).resolve().parents[1];run=market_run(root/'configs/basic_buyer.yaml',root=root)
        adapter=run.models[run.buyers[0].model].lora.path
        class FailedClient:
            def __init__(self,**kwargs):pass
            def complete(self,**kwargs):
                return SimpleNamespace(tool_call=None,stop_reason='error',outcome='infra_error',error='HTTP 400',text='')
        with tempfile.TemporaryDirectory(prefix='adl_bad_transport_cpu_',dir='/tmp') as temporary:
            folder=Path(temporary)
            plan={'devices':[{'index':'0','fraction':.8}],'port':8000,'seeds':[0,1],
                  'personas':{'operations_p1':personas()[0]},'adapters':{'base':None,'candidate':adapter}}
            (folder/'plan.json').write_text(json.dumps(plan));(folder/'cases.json').write_text(json.dumps([self.case()]))
            with patch('adlmarket.serving.VLLMCluster',return_value=nullcontext()), \
                 patch('adlmarket.serving.register_adapters'),patch('adlmarket.client.ModelClient',FailedClient):
                with self.assertRaisesRegex(RuntimeError,'evaluation invalid'):
                    worker(folder)
            self.assertEqual(json.loads((folder/'evaluation_status.json').read_text())['status'],'invalid')
            self.assertFalse((folder/'scoreboard.json').exists())

    def test_service_failure_invalidates_scores_but_model_no_tool_is_a_real_result(self):
        from training.diagnostic_scores import write_scores
        with tempfile.TemporaryDirectory(prefix='adl_score_validity_cpu_',dir='/tmp') as temporary:
            folder=Path(temporary);case=self.case()
            (folder/'cases.json').write_text(json.dumps([case]));(folder/'plan.json').write_text(json.dumps({'calls':1}))
            for outcome,valid in [('infra_error',False),('accepted',True)]:
                row={'case':case['name'],'context':'recorded','seed':0,'persona':'base/operations_p1',
                     'tool':None,'arguments':{},'outcome':outcome}
                (folder/'decisions.jsonl').write_text(json.dumps(row)+'\n')
                scores=write_scores(folder)
                self.assertEqual(scores['valid_evaluation'],valid)
                self.assertEqual('INVALID EVALUATION' in (folder/'scoreboard.md').read_text(),not valid)

    def test_zero_errors_from_waiting_must_not_be_called_improvement(self):
        from training.diagnostic_scores import write_scores
        from training.operation_gate import compare
        record=make_record(13,'binding_offer',0)
        case={**record,'name':record['id'],'contexts':['recorded'],'state_mode':'operations'}
        fn=record['answer']['tool_calls'][0]['function']
        self.assertEqual(fn['name'],'accept_offer')
        with tempfile.TemporaryDirectory(prefix='adl_wait_regression_cpu_',dir='/tmp') as temporary:
            folder=Path(temporary)
            (folder/'cases.json').write_text(json.dumps([case]));(folder/'plan.json').write_text(json.dumps({'calls':2}))
            common={'case':case['name'],'context':'recorded','seed':0,'gpu':'0','input_sha256':'identical-input',
                    'outcome':'accepted'}
            records=[{**common,'persona':'base/operations_p1','tool':fn['name'],'arguments':fn['arguments']},
                     {**common,'persona':'candidate/operations_p1','tool':'wait','arguments':{
                         'private_state':json.dumps({'bad':[],'pending':None,'last_buy':None})}}]
            (folder/'decisions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            write_scores(folder)
            result=compare(folder)
            self.assertEqual(result['personas']['operations_p1']['finding'],'lost_all_captures_by_waiting')


if __name__=='__main__':unittest.main()
