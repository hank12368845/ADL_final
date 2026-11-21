"""No real GPU query, sleeping or model launch in these tests."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
import io

from training.queue_seller_personas import wait_ready,main,launch


def row(i,free):return dict(index=str(i),uuid=f'cpu{i}',total_gib=24.,free_gib=free)


class QueueTests(unittest.TestCase):
    def test_waits_then_uses_one_without_waiting_for_both(self):
        frames=iter([[row(0,12),row(1,12)],[row(0,12),row(1,23)],[row(0,12),row(1,23)]])
        pause=[];messages=[]
        selected=wait_ready(['0','1'],2,8100,query=lambda:next(frames),pause=pause.append,
                            clock=lambda:0,port_check=lambda p:True,emit=messages.append)
        self.assertEqual(selected,['1']);self.assertEqual(pause,[15,15])
        self.assertIn('ETA unknown',messages[0])

    def test_prefers_both_when_available_but_can_force_one(self):
        kwargs=dict(query=lambda:[row(0,22),row(1,23)],pause=lambda s:None,
                    clock=lambda:0,port_check=lambda p:True,emit=lambda s:None)
        self.assertEqual(wait_ready(['0','1'],2,8100,**kwargs),['0','1'])
        self.assertEqual(wait_ready(['0','1'],1,8100,**kwargs),['1'])

    def test_explicit_gpu1_never_uses_gpu0(self):
        kwargs=dict(query=lambda:[row(0,23),row(1,22)],pause=lambda s:None,
                    clock=lambda:0,port_check=lambda p:True,emit=lambda s:None)
        self.assertEqual(wait_ready(['1'],2,8100,**kwargs),['1'])

    def test_busy_port_delays_launch(self):
        ports=iter([False,True,True]);pauses=[]
        selected=wait_ready(['1'],1,8100,query=lambda:[row(1,23)],pause=pauses.append,
                            clock=lambda:0,port_check=lambda p:next(ports),emit=lambda s:None)
        self.assertEqual(selected,['1']);self.assertEqual(len(pauses),2)

    def test_prepare_only_no_query_or_launch(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            with patch('sys.argv',['queue','--prepare-only','--output-dir',tmp]),patch('training.queue_seller_personas.query_gpus') as query,patch('training.queue_seller_personas.launch') as launch,redirect_stdout(io.StringIO()):
                folder=main();query.assert_not_called();launch.assert_not_called()
            self.assertTrue((folder/'compare_plan.json').exists())

    def test_requeues_memory_race_but_stops_other_errors(self):
        import json
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            folder=Path(tmp);first=folder/'v1';first.mkdir()
            marker=first/'wrapper_result.json';marker.write_text(json.dumps(dict(exit=0,actions=450)))
            original=marker.read_bytes()
            (folder/'compare_plan.json').write_text(json.dumps(dict(mock=False,arms={'v1':str(first),'v2':str(folder/'v2')})))
            with patch('sys.argv',['queue','--resume',tmp]),patch('training.queue_seller_personas.wait_ready',side_effect=[['0'],['1']]),patch('training.queue_seller_personas.launch',side_effect=[(1,True),(0,False)]) as run,redirect_stdout(io.StringIO()):
                main();self.assertEqual(run.call_count,2)
                self.assertEqual(run.call_args_list[0].args[1],['0']);self.assertEqual(run.call_args_list[1].args[1],['1'])
            self.assertEqual(marker.read_bytes(),original)
            with patch('sys.argv',['queue','--resume',tmp]),patch('training.queue_seller_personas.wait_ready',return_value=['1']),patch('training.queue_seller_personas.launch',return_value=(2,False)) as run,redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'exit2'):main()
                self.assertEqual(run.call_count,1)

    def test_launch_passes_only_selected_cards_to_comparison(self):
        from unittest.mock import Mock
        proc=Mock(stdout=io.StringIO('RuntimeError: No GPU has enough free memory after reserve\n'))
        proc.wait.return_value=1
        with patch('training.queue_seller_personas.subprocess.Popen',return_value=proc) as spawn:
            code,requeue=launch(Path('/tmp/example_queue'),['1'],8100,lambda text:None)
        command=spawn.call_args.args[0]
        self.assertEqual(command[command.index('--gpus')+1],'1')
        self.assertEqual(command[command.index('--port')+1],'8100')
        self.assertEqual((code,requeue),(1,True))


if __name__=='__main__':unittest.main()
