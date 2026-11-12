import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from scripts.validation_recovery import complete_checkpoint_validation,validation_complete

class ValidationRecoveryTests(unittest.TestCase):
    tasks=[dict(uid='fixed',seed=123)]
    cfg=dict(eval_every=25,steps=400)
    def complete(self,path):
        path.write_text(json.dumps(dict(n_tasks=1,n_trajectories=4,success_rate_pct=50.,pass_at_4_pct=100.)))
        path.with_suffix('.jsonl').write_text('\n'.join(json.dumps(dict(task=self.tasks[0],lineage=str(i),reward=i%2)) for i in range(4))+'\n')
    def test_missing_final_checkpoint_validation_is_replayed_once(self):
        with tempfile.TemporaryDirectory() as d:
            call=Mock(side_effect=self.complete)
            self.assertTrue(complete_checkpoint_validation(d,400,self.cfg,self.tasks,call))
            self.assertFalse(complete_checkpoint_validation(d,400,self.cfg,self.tasks,call))
            self.assertEqual(call.call_count,1)
    def test_partial_or_wrong_task_records_preserved_before_retry(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'validation_0050.json';self.complete(p)
            raw=p.with_suffix('.jsonl');original=raw.read_text().replace('fixed','wrong');raw.write_text(original)
            self.assertFalse(validation_complete(p,self.tasks))
            self.assertTrue(complete_checkpoint_validation(d,50,self.cfg,self.tasks,self.complete))
            backup=list((Path(d)/'validation_recovery_archive').glob('*/validation_0050.jsonl'))
            self.assertEqual(len(backup),1);self.assertEqual(backup[0].read_text(),original)
            self.assertTrue(validation_complete(p,self.tasks))
    def test_non_validation_checkpoint_does_not_add_eval_or_use_test(self):
        with tempfile.TemporaryDirectory() as d:
            call=Mock()
            self.assertFalse(complete_checkpoint_validation(d,40,self.cfg,self.tasks,call))
            call.assert_not_called()


def test_resume_at_full_queue_target_finishes_pending_test_without_update(tmp_path, monkeypatch):
    import random
    import sys
    import numpy as np
    import train
    from scripts import hardware, q35_thinking_policy

    cfg = dict(method='grpo', seed=42, steps=400, environment='sokoban',
               eval_every=25, learning_rate=1e-6)
    launch = tmp_path/'launch.json'
    launch.write_text(json.dumps(cfg))
    state = dict(config=cfg, step=400, model={}, optimizer={},
                 rng=np.random.default_rng(42).bit_generator.state,
                 python_rng=random.getstate(), torch_rng=train.torch.get_rng_state(), cuda_rng=[])
    train.torch.save(state, tmp_path/'checkpoint.pt')
    monkeypatch.setattr(sys, 'argv', ['train.py', '--config', str(launch), '--method', 'grpo',
                                      '--output-dir', str(tmp_path), '--resume', '--stop-after-step', '400'])
    monkeypatch.setattr(train, 'build_models', lambda *args: (Mock(), None, None, Mock()))
    monkeypatch.setattr(train.torch.optim, 'AdamW', Mock())
    monkeypatch.setattr(train, 'sync_weights', Mock())
    monkeypatch.setattr(train, 'update', Mock(side_effect=AssertionError('unexpected update')))
    monkeypatch.setattr(hardware, 'annotate_hardware', Mock())
    monkeypatch.setattr(q35_thinking_policy, 'check_prompt_contract', lambda *args: {})
    monkeypatch.setattr(train.signal, 'signal', Mock())
    monkeypatch.setenv('PATR_PROGRESS_PATH', '')
    tasks = ValidationRecoveryTests.tasks
    ValidationRecoveryTests().complete(tmp_path/'validation_0400.json')
    monkeypatch.setattr(train, 'make_datasets', lambda cfg: dict(validation=tasks, test=tasks))
    evaluator = Mock(return_value=dict(n_tasks=1))
    monkeypatch.setattr(train, 'evaluate', evaluator)
    train.main()
    assert evaluator.call_count == 1 and evaluator.call_args.args[-1] == 'test'
    assert json.loads((tmp_path/'status.json').read_text())['status'] == 'complete'
