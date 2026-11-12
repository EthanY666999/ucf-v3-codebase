import copy
from types import SimpleNamespace as NS
import numpy as np
import patr_core as core
FENCE=999
MAP={10:1,11:1,20:2,21:2,30:3,31:3,40:4,41:4}
class Tok:
 eos_token_id=2
 def encode(self,text,**kw):
  if text=='```':return [FENCE]
  word=text[3:-3];return [FENCE,{'Left':10,'Down':20,'Right':30,'Up':40}[word.strip()]+word.startswith(' '),FENCE]
 def decode(self,ids,**kw):
  table={FENCE:'```',2:'',90:'reason ',91:'more ',10:'Left',11:' Left',20:'Down',21:' Down',30:'Right',31:' Right',40:'Up',41:' Up'}
  return ''.join(table.get(x,'?') for x in ids)
def fixture():
 env=NS(desc=np.array([list('SF'),list('FG')],dtype='S1'),s=0,nrow=2,ncol=2)
 pre=core.Branch(0,env,NS(_parse_model_response=lambda text:('','1' if text.endswith('```Left```') else '3')),[100],[0],[0.],lineage='0')
 b=copy.deepcopy(pre);b.ids += [90,91,FENCE,30,FENCE,2];b.mask += [1]*6;b.old_logprobs += [-.3,-.4,-.2,-1.,-.1,-.1]
 b.trace=[{'response':'reason more ```Right```'}]
 n=dict(branch=pre,j=0,turn=1,state=0,boundary=1,S=.3,parent_action=3)
 return pre,b,n
