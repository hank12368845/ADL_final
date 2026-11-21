"""CPU regression checks for endpoint deduplication and experiment equivalence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from adlmarket.setups import market_run
from training.local_resources import gpu_models
from training.market_load_balance import balance_ta_routes, profile_costs, validate_equivalent

ROOT = Path(__file__).resolve().parents[1]


class LoadBalanceTests(unittest.TestCase):
    def test_same_ta_weights_can_reach_both_servers_without_changing_agents(self):
        source = ROOT / 'configs/buyer_scored_short.yaml'
        original = market_run(source, root=ROOT)
        config = yaml.safe_load(source.read_text())
        devices = [{'index': '0', 'fraction': .8}, {'index': '1', 'fraction': .7}]
        models = gpu_models(yaml.safe_load((ROOT / 'configs/models.yaml').read_text()), devices, 8100)
        with tempfile.TemporaryDirectory(prefix='adl_route_test_', dir='/tmp') as temporary:
            folder = Path(temporary)
            plan = balance_ta_routes(config, original, models, folder, ROOT, devices, 8100)
            (folder / 'models.yaml').write_text(yaml.safe_dump(models))
            config['models'] = str(folder / 'models.yaml')
            (folder / 'input.yaml').write_text(yaml.safe_dump(config))
            after = market_run(folder / 'input.yaml', root=ROOT)
            validate_equivalent(original, after)
            routes = {a.id: after.models[a.model].url_for(after.models[a.model].model)
                      for a in after.everyone()}
            for aid, device in plan['assignments'].items():
                self.assertEqual(routes[aid], f'http://localhost:{8100+device}/v1')
            # Same TA buyer adapter must actually be usable on BOTH endpoints.
            self.assertEqual({routes[a.id] for a in after.buyers if a.owner == 'ta'},
                             {'http://localhost:8100/v1', 'http://localhost:8101/v1'})
            for item in plan['adapter_copies']:
                src = Path(item['source']) / 'adapter_model.safetensors'
                dst = Path(item['destination']) / 'adapter_model.safetensors'
                self.assertEqual(src.read_bytes(), dst.read_bytes())
            changed = after.model_copy(deep=True)
            changed.buyers[0].reasoning = not changed.buyers[0].reasoning
            with self.assertRaisesRegex(ValueError, 'behavior'):
                validate_equivalent(original, changed)

    def test_single_server_needs_no_adapter_copies(self):
        config, models = {'sellers': []}, {}
        before = copy.deepcopy(config)
        plan = balance_ta_routes(config, None, models, None, ROOT, [{'index': '0'}], 8000)
        self.assertEqual(plan['method'], 'single_server')
        self.assertEqual(config, before)

    def test_mock_cost_profile_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='adl_profile_test_', dir='/tmp') as temporary:
            folder = Path(temporary)
            (folder / 'metadata.json').write_text(json.dumps({'mock_llm': True}))
            with self.assertRaisesRegex(ValueError, 'mock'):
                profile_costs(folder)


if __name__ == '__main__':
    unittest.main()
