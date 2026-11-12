import json
import tempfile
import unittest
from pathlib import Path
from scripts.ucf_recovery import reconcile_ucf

class UCFRecoveryTests(unittest.TestCase):
    def test_restore_checkpoint_preserve_later_evidence_and_allow_replay(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            committed=dict(last_step=10,c=0.7,history=[1.5]*10)
            ahead=dict(last_step=12,c=0.9,history=[1.5]*10+[2,3])
            (p/'controller_0010.json').write_text(json.dumps(committed))
            (p/'controller.json').write_text(json.dumps(ahead))
            for name in ['controller_0012.json','select_0011.jsonl','budget_0011.json','raw_leaves_0012.jsonl.gz']:
                (p/name).write_bytes(b'preserve failed attempt')
            (p/'select_0010.jsonl').write_text('committed evidence')
            result=reconcile_ucf(p,10)
            self.assertEqual(json.loads((p/'controller.json').read_text()),committed)
            self.assertEqual((p/'select_0010.jsonl').read_text(),'committed evidence')
            archived=Path(result['archive'])
            self.assertEqual(json.loads((archived/'controller.json').read_text()),ahead)
            self.assertEqual((archived/'raw_leaves_0012.jsonl.gz').read_bytes(),b'preserve failed attempt')
            self.assertFalse(reconcile_ucf(p,10)['changed'])
            # UCF durable selection uses exclusive create: replay no longer collides.
            with (p/'select_0011.jsonl').open('x') as f:f.write('replayed selection')
    def test_missing_snapshot_rejects_before_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'controller.json').write_text('original')
            (p/'select_0011.jsonl').write_text('evidence')
            with self.assertRaises(AssertionError):reconcile_ucf(p,10)
            self.assertEqual((p/'controller.json').read_text(),'original')
            self.assertTrue((p/'select_0011.jsonl').exists())
            self.assertFalse((p/'recovery_archive').exists())
    def test_step_zero_restores_fresh_controller(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'controller.json').write_text('failed first attempt')
            result=reconcile_ucf(p,0)
            self.assertFalse((p/'controller.json').exists())
            self.assertEqual((Path(result['archive'])/'controller.json').read_text(),'failed first attempt')
            self.assertFalse(reconcile_ucf(p,0)['changed'])
