import unittest,copy
from types import SimpleNamespace as NS
import numpy as np
from patr_core import Branch,training_weights
from external_baselines import gigpo_weights,starpo_select,configure

def trajectories(task=0):
 env=NS(desc=np.array([list('SFF'),list('FFG')],dtype='S1'))
 return [Branch(task,env,None,[99,1,2,10,3,4],[0,1,1,0,1,1],[0,-.1,-.1,0,-.2,-.2],lineage=str(j),reward=r,trace=[dict(action=3,state=s),dict(action=2,state=5)]) for j,(r,s) in enumerate([(1,1),(0,1),(1,3),(0,4)])]
class BaselineTests(unittest.TestCase):
 def test_gigpo_hand_calculation(self):
  bs=trajectories();rows=gigpo_weights(bs,1);eps=1e-6
  # Episode returns [1,0,1,0], sample std sqrt(1/3).
  a=.5/(np.sqrt(1/3)+eps)
  # First anchor uses discounted returns [.95,0,.95,0].
  first=.475/(.95*np.sqrt(1/3)+eps)
  # Second anchor at position 1 has two members [1,0]; singleton groups give zero.
  second=.5/(np.sqrt(.5)+eps)
  expected=[a+first,a+second,-a-first,-a-second,a+first,a,-a-first,-a]
  np.testing.assert_allclose([x[1] for x in rows],expected,rtol=0,atol=1e-14)
  self.assertEqual([x[0].gigpo['group_size'] for x in rows],[4,2,4,2,4,1,4,1])
  for b,adv,w in rows:self.assertEqual(b.ntokens,2);self.assertEqual(w,1/16)
  self.assertEqual(sum(b.ntokens*w for b,_,w in rows),1.)
 def test_omega_zero_exact_grpo(self):
  bs=trajectories();self.assertEqual(gigpo_weights(bs,1,omega=0),training_weights(bs,1))
 def test_task_anchor_isolation(self):
  bs=trajectories(0)+trajectories(1)
  rows=gigpo_weights(bs,2)
  self.assertTrue(all(b.gigpo['group_size']<=4 for b,_,_ in rows))
 def test_starpo_top_reward_std_and_ties(self):
  bs=[]
  for i,rewards in enumerate([[0,0,0,0],[0,1,0,1],[1,1,1,0],[1,1,1,1]]):
   group=trajectories(i)
   for b,r in zip(group,rewards):b.reward=r
   bs+=group
  selected,keep,scores=starpo_select(bs,4,.5)
  self.assertEqual(keep,[1,2]);self.assertEqual(len(selected),8)
  self.assertEqual({b.task_id for b in selected},{0,1})
  self.assertEqual(bs[4].task_id,1)
 def test_starpo_configuration_and_missing_protocol(self):
  train=NS(update=lambda *args:None);cfg={'starpo_keep_fraction':.5,'kl_coefficient':.001,'clip_low':.28,'clip_high':.28}
  configure(train,cfg,'starpo_s')
  self.assertEqual((cfg['kl_coefficient'],cfg['clip_low'],cfg['clip_high']),(0.,.2,.28))
  with self.assertRaises(ValueError):configure(NS(update=lambda *args:None),{},'starpo_s')
