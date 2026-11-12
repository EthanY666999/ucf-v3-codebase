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
