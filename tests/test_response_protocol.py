import ast,copy,gzip,json,subprocess,sys,unittest
from pathlib import Path
from types import SimpleNamespace as NS
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import patr_core as core
from response_protocol import NAME,first_block
class ActualProtocol(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  from transformers import AutoTokenizer
  cls.tok=AutoTokenizer.from_pretrained(ROOT/'metadata/Qwen3.5-2B',local_files_only=True)
  cls.cfg=dict(json.loads((ROOT/'configs/common.json').read_text()),**{'max_turns':10})
  task=json.loads((ROOT/'data/sokoban/data.json').read_text())['train'][0]
  cls.row={'task':task}
 def out(self,text,reason='stop'):
  ids=self.tok.encode(text,add_special_tokens=False)+[self.tok.eos_token_id]
  return NS(token_ids=ids,logprobs=[{t:NS(logprob=-.1)} for t in ids],text=text,finish_reason=reason)
 def branch(self,cfg):return core.start_branch(self.row['task'],0,self.tok,cfg,'0')
 def view(self,b):return {k:getattr(b,k) for k in ['ids','mask','old_logprobs','actions','trace','reward','turns','terminal']}
 def test_first_block_end_executed_even_if_unused_tail_hit_length(self):
  cfg=dict(self.cfg,response_protocol=NAME);b=self.branch(cfg);prompt=len(b.ids)
  out=self.out('draft ``` Right ``` later ```Left```','length')
  core.apply_response(b,out,self.tok,cfg)
  keep=self.tok.encode('draft ``` Right ```',add_special_tokens=False)
  self.assertEqual(b.ids[prompt:prompt+len(keep)],keep)
  self.assertEqual(b.mask[prompt:prompt+len(keep)],[1]*len(keep))
  self.assertEqual(b.ids[prompt+len(keep)],self.tok.eos_token_id);self.assertEqual(b.mask[prompt+len(keep)],0)
  self.assertEqual(b.actions,[3]);self.assertEqual(b.trace[0]['response'],'draft ``` Right ```')
  self.assertEqual(sum(b.mask),len(keep));self.assertEqual(len(b.ids),len(b.old_logprobs))
 def test_unfinished_first_draft_does_not_hide_first_complete_block(self):
  cfg=dict(self.cfg,response_protocol=NAME);b=self.branch(cfg)
  core.apply_response(b,self.out('draft ``` Right </think> now ```Down``` later ```Left```'),self.tok,cfg)
  self.assertEqual(b.actions,[2]);self.assertTrue(b.trace[0]['response'].endswith('```Down```'))
 def test_atomic_closing_token_retained_without_retokenizing(self):
  cfg=dict(self.cfg,response_protocol=NAME);b=self.branch(cfg);prompt=len(b.ids)
  out=self.out('Output ```Up```. discarded ```Down```')
  core.apply_response(b,out,self.tok,cfg)
  n=sum(b.mask);self.assertEqual(b.ids[prompt:prompt+n],out.token_ids[:n])
  self.assertEqual(b.actions,[4]);self.assertEqual(b.trace[0]['response'],'Output ```Up```.')
  self.assertEqual(b.trace[0]['protocol_cutoff']['boundary_token_overhang'],'.')
  self.assertEqual(b.old_logprobs[prompt:prompt+n],[-.1]*n)
  from response_protocol import protocol_metrics
  report=protocol_metrics([b]);self.assertEqual(report['boundary_token_extra_fraction'],1.)
 def test_no_complete_block_retains_invalid_or_length_behavior(self):
  cfg=dict(self.cfg,response_protocol=NAME)
  for reason in ['stop','length']:
   b=self.branch(cfg);out=self.out('```unfinished',reason);core.apply_response(b,out,self.tok,cfg)
   self.assertNotIn('protocol_cutoff',b.trace[0]);self.assertEqual(b.trace[0].get('action'),None if reason=='length' else 0)
if __name__=='__main__':unittest.main()
