import os,time

def requests(engine,jobs):
 backend=engine.llm_engine;serial=getattr(engine,'_patr_request_serial',0);pending={};result={};cost=dict(input_tokens=0,generated_tokens=0,request_seconds=0.)
 start=time.monotonic();last_progress=0.
 for key,ids,params in jobs:
  rid=f'ucf-{serial}';serial+=1;backend.add_request(rid,{'prompt_token_ids':ids},params);pending[rid]=(key,time.monotonic());cost['input_tokens']+=len(ids)
 engine._patr_request_serial=serial
 while pending:
  if time.monotonic()-last_progress>=5 and os.environ.get('PATR_PROGRESS_PATH'):
   from patr_core import atomic_json
   atomic_json(os.environ['PATR_PROGRESS_PATH'],dict(phase='ucf_probe_or_forced_scoring',active=len(pending),completed=len(result),generated_tokens=cost['generated_tokens'],updated_unix=time.time()))
   last_progress=time.monotonic()
  for out in backend.step():
   if not out.finished:continue
   key,ts=pending.pop(out.request_id);result[key]=out;cost['generated_tokens']+=sum(len(x.token_ids) for x in out.outputs);cost['request_seconds']+=time.monotonic()-ts
 cost['seconds']=time.monotonic()-start
 return result,cost
