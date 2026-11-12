import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from scripts.q35_engine_loader import engine_source,build_engine

class EngineLoader(unittest.TestCase):
 def fixture(self,tmp):
  base=Path(tmp)/'base';saved=Path(tmp)/'saved';base.mkdir();saved.mkdir()
  text=dict(model_type='qwen3_5_text',hidden_size=8,vocab_size=16,use_cache=True,dtype='bfloat16')
  (base/'config.json').write_text(json.dumps(dict(model_type='qwen3_5',text_config=text)))
  (saved/'config.json').write_text(json.dumps(dict(text,dtype='float32',architectures=['Qwen3_5ForCausalLM'],transformers_version='5',use_cache=False,bos_token_id=None,pad_token_id=None)))
  return base,saved
 def test_export_syncs_trained_weights_before_return(self):
  with tempfile.TemporaryDirectory() as tmp:
   base,saved=self.fixture(tmp);calls=[]
   def factory(**kw):
    self.assertEqual(kw['model'],str(base));self.assertEqual(kw['tokenizer'],str(saved))
    calls.append('create');return SimpleNamespace(weight=0)
   def sync(actor,engine):calls.append('sync');engine.weight=actor.weight
   e=build_engine(SimpleNamespace(weight=7),saved,factory,sync,base=base)
   self.assertEqual(e.weight,7);self.assertEqual(calls,['create','sync'])
 def test_original_base_path_unchanged(self):
  with tempfile.TemporaryDirectory() as tmp:
   base,saved=self.fixture(tmp);calls=[]
   e=build_engine(None,base,lambda **kw:kw,lambda *a:calls.append(a),base=base)
   self.assertEqual(e['model'],str(base));self.assertEqual(calls,[])
 def test_architecture_mismatch_rejected_before_allocation(self):
  with tempfile.TemporaryDirectory() as tmp:
   base,saved=self.fixture(tmp);d=json.loads((saved/'config.json').read_text());d['hidden_size']=12
   (saved/'config.json').write_text(json.dumps(d))
   with self.assertRaisesRegex(AssertionError,'architecture differs'):build_engine(None,saved,lambda **kw:self.fail('allocated'),None,base=base)
 def test_missing_bootstrap_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   base,saved=self.fixture(tmp)
   with self.assertRaises(FileNotFoundError):engine_source(saved,base/'missing')
