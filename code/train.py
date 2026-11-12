import os
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'  # V1 engine, kept in-process so weights can be synced directly
os.environ['VLLM_USE_FLASHINFER_SAMPLER']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['OMP_NUM_THREADS']='8'
os.environ['VLLM_NO_USAGE_STATS']='1'
os.environ['DO_NOT_TRACK']='1'
os.environ['WANDB_MODE']='offline'

import argparse
import copy
import gzip
import hashlib
import json
import random
import signal
import shutil
from functools import partial
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.release_config import check_release
from patr_core import (ROOT, atomic_json, append_json, make_datasets, collect,
                       evaluation_metrics, training_weights)

STOP=False
RUNTIME_KEYS={'micro_batch_size','micro_batch_token_limit','vllm_memory_fraction','vllm_max_sequences','vllm_enforce_eager','vllm_attention_backend'}


def request_stop(signum, frame):
    global STOP
    STOP=True


def build_models(cfg,seed):
    backend=cfg.get('vllm_attention_backend','FLASH_ATTN')
    assert backend in {'FLASH_ATTN','XFORMERS'},'Unsupported inference attention backend'
    # Retain backend in recorded configuration; vLLM selects its supported backend.
    # Triton builds its Python extension on first use. Minimal host images may
    # omit Python.h; matching distro headers can be extracted without root.
    local_include=ROOT/'artifacts/build-deps/usr/include'
    if (local_include/'python3.12/Python.h').exists():
        include_paths=[str(local_include/'python3.12'),str(local_include)]
        if os.environ.get('C_INCLUDE_PATH'):include_paths.append(os.environ['C_INCLUDE_PATH'])
        os.environ['C_INCLUDE_PATH']=os.pathsep.join(include_paths)
    from vllm import LLM
    torch.set_num_threads(8)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    model_path=str(ROOT/cfg['model'])
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token
    # Load actor/ref before LLM: the reverse order triggers Qwen3_5Config.vocab_size errors.
    actor=AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.float32,
                   attn_implementation='sdpa',local_files_only=True).to(cfg.get('actor_device','cuda:0'))
    actor.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    actor.config.use_cache=False
    ref=AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.bfloat16,
                   attn_implementation='sdpa',local_files_only=True).to('cuda:0').eval()
    ref.requires_grad_(False)
    from scripts.q35_engine_loader import build_engine
    engine=build_engine(actor,model_path,LLM,sync_weights,dtype='bfloat16',
               tensor_parallel_size=1,distributed_executor_backend='uni',
               gpu_memory_utilization=cfg['vllm_memory_fraction'],
               max_model_len=cfg['max_context_tokens'],max_num_seqs=cfg['vllm_max_sequences'],
               max_num_batched_tokens=4096,enable_chunked_prefill=True,
               enable_prefix_caching=True,limit_mm_per_prompt={'image':0,'video':0},
               seed=seed,enforce_eager=cfg.get('vllm_enforce_eager',False),disable_log_stats=True,
               max_logprobs=20)
    return actor,ref,engine,tokenizer


def sync_weights(actor,engine):
    # V1 in-process engine colocates the inference model. No network service or extra GPU.
    vm=engine.llm_engine.model_executor.driver_worker.model_runner.model
    # vLLM loads multimodal Qwen3.5 as ...ForConditionalGeneration: HF text-only names
    # `model.*` must be presented as `model.language_model.*`; tied lm_head is skipped.
    text=getattr(vm,'language_model',None)
    tied=getattr(actor.config,'tie_word_embeddings',False)
    inference_device = next(vm.parameters()).device
    def named():
        for name,p in actor.named_parameters():
            if tied and name=='lm_head.weight':continue
            if text is not None and name.startswith('model.'):name='model.language_model.'+name[len('model.'):]
            yield name,p.detach().to(device=inference_device, dtype=torch.bfloat16)
    with torch.no_grad():
        loaded=vm.load_weights(named())
        vlayers=(text.model if text is not None else vm.model).layers
        assert torch.equal(vlayers[-1].mlp.down_proj.weight,
                           actor.model.layers[-1].mlp.down_proj.weight.to(device=inference_device, dtype=torch.bfloat16)), 'Inference weight synchronization failed'
    engine.llm_engine.reset_prefix_cache()
    torch.cuda.synchronize()
    return len(loaded) if loaded is not None else None


def selected_logprobs(model,ids,attention,mask,gradient=False):
    """Only allocate vocabulary logits for assistant targets, in bounded chunks."""
    caller_device = ids.device
    device = next(model.parameters()).device
    ids, attention, mask = (x.to(device) for x in (ids, attention, mask))
    with torch.autocast('cuda',dtype=torch.bfloat16):
        hidden=model.model(input_ids=ids,attention_mask=attention,use_cache=False).last_hidden_state
        keep=mask[:,1:].bool()
        h=hidden[:,:-1][keep]
        labels=ids[:,1:][keep]
        parts=[]
        def head_logprob(x,y):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model.lm_head(x)
            return -F.cross_entropy(logits.float(),y,reduction='none')
        for lo in range(0,len(labels),512):
            hs,ys=h[lo:lo+512],labels[lo:lo+512]
            lp=checkpoint(head_logprob,hs,ys,use_reentrant=False) if gradient else head_logprob(hs,ys)
            parts.append(lp)
        return torch.cat(parts).to(caller_device)


def pack(rows,pad,device='cuda:0'):
    maxlen=max(len(b.ids) for b,_,_ in rows)
    ids=torch.full((len(rows),maxlen),pad,device=device,dtype=torch.long)
    attn=torch.zeros_like(ids)
    mask=torch.zeros_like(ids)
    advantages=[];weights=[];old=[]
    for i,(b,a,w) in enumerate(rows):
        n=len(b.ids)
        ids[i,:n]=torch.tensor(b.ids,device=device)
        attn[i,:n]=1
        mask[i,:n]=torch.tensor(b.mask,device=device)
        advantages.extend([a]*b.ntokens)
        weights.extend([w]*b.ntokens)
        old.extend(v for v,m in zip(b.old_logprobs,b.mask) if m)
    f=lambda x:torch.tensor(x,dtype=torch.float32,device=device)
    return ids,attn,mask,f(advantages),f(weights),f(old)


def microbatches(rows,cfg):
    # Sort only within the current optimizer minibatch to reduce padding.
    rows=sorted(rows,key=lambda x:len(x[0].ids))
    batch=[]
    for row in rows:
        if batch and (len(batch)>=cfg['micro_batch_size'] or
                      len(row[0].ids)*(len(batch)+1)>cfg['micro_batch_token_limit']):
            yield batch
            batch=[]
        batch.append(row)
    if batch:
        yield batch


def update(actor,ref,optimizer,branches,cfg,method,tokenizer,rng,n_tasks):
    actor.train()
    metrics=Counter();mb_count=0
    task_order=rng.permutation(n_tasks).tolist()
    for off in range(0,n_tasks,cfg['mini_batch_tasks']):
        chosen=task_order[off:off+cfg['mini_batch_tasks']]
        idmap={old:new for new,old in enumerate(chosen)}
        # Only remap a shallow copy; rollout records retain their original task IDs.
        bs=[]
        for b in branches:
            if b.task_id in idmap:
                c=copy.copy(b);c.task_id=idmap[b.task_id];bs.append(c)
        rows=training_weights(bs,len(chosen))
        optimizer.zero_grad(set_to_none=True)
        low=cfg['clip_low']
        total_loss=0.;before=None
        for chunk in microbatches(rows,cfg):
            ids,attn,mask,adv,weight,old=pack(chunk,tokenizer.pad_token_id,next(actor.parameters()).device)
            with torch.no_grad():
                ref_lp=selected_logprobs(ref,ids,attn,mask)
            current=selected_logprobs(actor,ids,attn,mask,gradient=True)
            ratio=torch.exp(current-old)
            pg=-torch.minimum(ratio*adv,torch.clamp(ratio,1-low,1+cfg['clip_high'])*adv)
            # Low-variance sampled KL against the frozen initial policy.
            logratio=ref_lp-current
            kl=torch.exp(logratio)-logratio-1
            loss=((pg+cfg['kl_coefficient']*kl)*weight).sum()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss; preserving previous checkpoint')
            loss.backward()
            total_loss+=loss.detach().item()
            metrics['kl']+=(kl.detach()*weight).sum().item()
            metrics['clipfrac']+=(((ratio.detach()<1-low)|(ratio.detach()>1+cfg['clip_high'])).float()*weight).sum().item()
            metrics['sampled_surprisal']+=((-current.detach())*weight).sum().item()
            metrics['weighted_response_tokens']+=len(old)
            del ids,attn,mask,adv,weight,old,ref_lp,current,ratio,pg,kl,loss,logratio
        norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),cfg['max_grad_norm'])
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite gradient norm')
        probe=actor.model.layers[-1].mlp.down_proj.weight
        before=probe.detach().clone()
        optimizer.step()
        delta=(probe.detach()-before).abs().max().item()
        metrics['max_parameter_change']=max(metrics['max_parameter_change'],delta)
        metrics['loss']+=total_loss
        metrics['grad_norm']+=float(norm)
        mb_count+=1
    optimizer.zero_grad(set_to_none=True)
    actor.eval()
    for key in ['loss','grad_norm','kl','clipfrac','sampled_surprisal']:
        metrics[key]/=mb_count
    return dict(metrics)


def save_checkpoint(path,actor,optimizer,step,rng,cfg,elapsed):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    # The old checkpoint remains intact if the temporary replacement cannot
    # fit. Reserve8GiB for logs and the operating system on this mount.
    tensor_bytes=sum(p.numel()*p.element_size() for p in actor.parameters())
    tensor_bytes+=sum(v.numel()*v.element_size() for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v))
    required=int(tensor_bytes*1.03)+8*2**30
    if shutil.disk_usage(path.parent).free<required:
        raise OSError(f'Checkpoint disk reserve: need {required} free bytes before atomic write')
    # torch.save handles CUDA tensors; restore maps to CPU before copying to model.
    state={'step':step,'model':actor.state_dict(),'optimizer':optimizer.state_dict(),
           'rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
           'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),
           'config':cfg,'elapsed_seconds':elapsed}
    tmp=path.with_suffix('.tmp')
    torch.save(state,tmp)
    tmp.replace(path)


def evaluate(engine,tokenizer,cfg,tasks,out_path,seed,split):
    rng=np.random.default_rng(seed)
    start=time.monotonic()
    branches,stats,_=collect(tasks,engine,tokenizer,cfg,'grpo',rng,evaluation=True)
    metrics=evaluation_metrics(branches,tasks)
    if cfg.get('response_protocol')=='first_complete_action_block':
        from response_protocol import protocol_metrics
        metrics['response_protocol_metrics']=protocol_metrics(branches)
    metrics.update(split=split,seconds=time.monotonic()-start,generated_tokens=stats['generated_tokens'])
    atomic_json(out_path,metrics)
    records=Path(out_path).with_suffix('.jsonl')
    with records.open('w') as f:
        for b in branches:
            f.write(json.dumps(dict(task=tasks[b.task_id],lineage=b.lineage,reward=b.reward,
                         terminal=b.terminal,turns=b.turns,response_tokens=b.ntokens,trace=b.trace,
                         **({'webshop_score': b.env.raw_score} if hasattr(b.env, 'raw_score') else {})))+'\n')
    return metrics


def flh_base_assessment(metrics, difficulty):
    success = metrics['success_rate_pct']
    pass4 = metrics['pass_at_4_pct']
    if not (0 <= success <= 100 and 0 <= pass4 <= 100):
        raise ValueError('Invalid step-0 validation percentages')
    if success < 5 or pass4 < 20:
        outcome = 'below_target'
    elif success > 20 or pass4 > 50:
        outcome = 'above_target'
    else:
        outcome = 'in_target'
    return dict(difficulty=difficulty, outcome=outcome, success_rate_pct=success, pass_at_4_pct=pass4,
                target_success_rate_pct=[5, 20], target_pass_at_4_pct=[20, 50])


def write_rollouts(path,branches,tasks,mode='wt',attempt=None):
    with gzip.open(path,mode) as f:
        for b in branches:
            f.write(json.dumps(dict(task=tasks[b.task_id],lineage=b.lineage,reward=b.reward,
                terminal=b.terminal,ids=b.ids,mask=b.mask,old_logprobs=b.old_logprobs,
                trace=b.trace,resample_attempt=attempt))+'\n')


def configure_ucf(cfg, out):
    import patr_core
    from ucf.ucf_v31 import make_collect
    from ucf.ucf_weights import training_weights_ucf
    from ucf.component_grad import component_grad, component_names

    global collect, training_weights, update
    collect = make_collect(patr_core, patr_core.collect)
    training_weights = partial(training_weights_ucf, cfg=cfg)
    original_update = update
    def ucf_update(actor, ref, optimizer, branches, cfg, method, tokenizer, rng, n_tasks):
        step = json.loads((out / 'status.json').read_text())['current_step']
        measured = step == 1 or step % cfg['eval_every'] == 0
        start = time.monotonic()
        diagnostics = {part: component_grad(actor, ref, branches, cfg, tokenizer, part)
                       for part in component_names(cfg)} if measured else None
        seconds = time.monotonic() - start
        result = original_update(actor, ref, optimizer, branches, cfg, method, tokenizer, rng, n_tasks)
        result.update(ucf_preupdate_fullbatch_components=diagnostics,
                      ucf_component_diagnostics_computed=measured,
                      ucf_component_diagnostic_seconds=seconds)
        return result
    update = ucf_update
    cfg['ucf_log_dir'] = str(out / 'ucf_logs')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',required=True,choices=['grpo','ucf','gigpo','starpo_s'])
    parser.add_argument('--config',required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after-step',type=int)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--eval-only',action='store_true')
    parser.add_argument('--validate-base',action='store_true')
    args=parser.parse_args()
    cfg=json.loads((ROOT/args.config).read_text())
    check_release(cfg, args.method)
    if args.validate_base and (args.smoke or args.resume or args.eval_only or
                               args.stop_after_step is not None):
        parser.error('--validate-base requires fresh weights and exits at step 0')
    if args.smoke:
        cfg.update(steps=1,train_batch_size=4,mini_batch_tasks=4,eval_every=1,save_every=1)
    seed=cfg['seed']
    out=args.output_dir or ROOT/'runs'/('smoke' if args.smoke else 'formal')/f'{args.method}_seed{seed}'
    cfg['method'] = args.method
    cp = out / 'checkpoint.pt'
    if args.resume:
        state = torch.load(cp, map_location='cpu', weights_only=False, mmap=True)
        check_release(state['config'], args.method)
    if args.method in ['starpo_s','gigpo']:
        import sys
        from external_baselines import configure
        configure(sys.modules[__name__],cfg,args.method)
    if args.method=='ucf':
        configure_ucf(cfg, out)
    if args.resume:
        previous = {k:v for k,v in state['config'].items() if k not in RUNTIME_KEYS}
        requested = {k:v for k,v in cfg.items() if k not in RUNTIME_KEYS}
        if requested != previous:
            raise ValueError('Cannot resume after an algorithm configuration change')
    out.mkdir(parents=True,exist_ok=True)
    from scripts.hardware import annotate_hardware
    annotate_hardware(out,cfg)
    source_files={name:(ROOT/name).read_bytes() for name in ('train.py','patr_core.py')}
    source_hash=hashlib.sha256(b''.join(source_files.values())).hexdigest()
    snapshot=out/'source_snapshots'/source_hash
    snapshot.mkdir(parents=True,exist_ok=True)
    for name,content in source_files.items():(snapshot/name).write_bytes(content)
    append_json(out/'source_versions.jsonl',dict(sha256=source_hash,unix=time.time(),resume=args.resume))
    atomic_json(out/'config.json',dict(cfg,method=args.method))
    atomic_json(out/'status.json',dict(status='initializing',method=args.method,step=0,total_steps=cfg['steps'],pid=os.getpid()))
    data=make_datasets(cfg)
    actor,ref,engine,tokenizer=build_models(cfg,seed)
    from scripts.q35_thinking_policy import check_prompt_contract
    atomic_json(out/'prompt_contract.json',check_prompt_contract(tokenizer))
    if args.eval_only:
        evaluate(engine,tokenizer,cfg,data['validation'],out/'validation_0000.json',100042,'validation')
        evaluate(engine,tokenizer,cfg,data['test'],out/'final_test.json',200042,'test')
        atomic_json(out/'status.json',dict(status='evaluation_complete',method=args.method))
        return
    # Record the original model's scores on the map set selected at launch.
    fresh_flh = cfg.get('environment') == 'flh' and not (args.resume or args.smoke)
    if args.validate_base or fresh_flh:
        os.environ['PATR_PROGRESS_PATH'] = str(out/'progress.json')
        base = evaluate(engine, tokenizer, cfg, data['validation'], out/'validation_0000.json', 100042, 'validation')
        print(json.dumps({'initial_validation': base}), flush=True)
        if fresh_flh:
            difficulty = cfg.get('flh_difficulty', 'hard')
            assessment = flh_base_assessment(base, difficulty)
            assessment.update(n_tasks=base['n_tasks'], n_trajectories=base['n_trajectories'],
                              max_response_tokens=cfg['max_response_tokens'],
                              dataset_sha256=cfg['dataset_sha256'], maps_sha256=cfg['frozen_maps_sha256'])
            atomic_json(out/'flh_calibration.json', assessment)
            print(json.dumps({'flh_calibration': assessment}), flush=True)
        if args.validate_base:
            atomic_json(out/'status.json', dict(status='base_validation_complete', step=0, method=args.method))
            return
    optimizer_name=cfg.get('optimizer','adamw')
    if optimizer_name=='adamw8bit':
        import bitsandbytes as bnb
        optimizer_cls=bnb.optim.AdamW8bit
    elif optimizer_name=='adamw':
        optimizer_cls=torch.optim.AdamW
    else:
        raise ValueError(f'Unsupported optimizer: {optimizer_name}')
    optimizer_options = {'foreach': False} if cfg.get('actor_device') == 'cuda:1' and optimizer_name == 'adamw' else {}
    optimizer=optimizer_cls(actor.parameters(),**optimizer_options,lr=cfg['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=0.)
    rng=np.random.default_rng(seed)
    start_step=0;previous_elapsed=0.
    if args.resume:
        actor.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        start_step=state['step'];rng.bit_generator.state=state['rng']
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);random.setstate(state['python_rng'])
        previous_elapsed=state.get('elapsed_seconds',0.)
        del state
        from scripts.run_recovery import archive_uncommitted
        archive_uncommitted(out,start_step)
        sync_weights(actor,engine)
        print(json.dumps({'RESUMED':start_step,'checkpoint':str(cp)}),flush=True)
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    os.environ['PATR_PROGRESS_PATH']=str(out/'progress.json')
    # Freeze test use to final evaluation. Training curves use a separate validation set.
    validation=data['validation'][:4] if args.smoke else data['validation']
    if start_step==0 and not fresh_flh:
        base=evaluate(engine,tokenizer,cfg,validation,out/'validation_0000.json',100042,'validation')
        print(json.dumps({'initial_validation':base}),flush=True)
    elif args.resume:
        from scripts.validation_recovery import complete_checkpoint_validation
        complete_checkpoint_validation(out,start_step,cfg,validation,
            lambda path:evaluate(engine,tokenizer,cfg,validation,path,100042,'validation'))
    # A restarted short pilot may already be at its requested boundary. Finish
    # pending checkpoint validation above, but never perform another update or
    # enter final-test evaluation merely because the supervisor restarted it.
    if args.resume and args.stop_after_step is not None and args.stop_after_step <= start_step < cfg['steps']:
        atomic_json(out/'status.json',dict(status='paused_checkpointed',step=start_step,
                    total_steps=cfg['steps'],method=args.method,
                    final_evaluation_pending=start_step==cfg['steps'],
                    stop_after_step=args.stop_after_step,reason='requested_stop_boundary'))
        return
    start=time.monotonic()
    step=start_step
    for step in range(start_step+1,cfg['steps']+1):
        if STOP:
            step-=1
            break
        atomic_json(out/'status.json',dict(status='training',method=args.method,step=step-1,
                      current_step=step,total_steps=cfg['steps'],pid=os.getpid()))
        t0=time.monotonic()
        # Keep the task stream independent of method-specific branching RNG consumption.
        task_rng=np.random.default_rng(np.random.SeedSequence([seed,step,1701]))
        task_indices=task_rng.choice(len(data['train']),cfg['train_batch_size'],replace=False).tolist()
        tasks=[data['train'][i] for i in task_indices]
        branches,stats,events=collect(tasks,engine,tokenizer,cfg,'grpo' if args.method in ['starpo_s','gigpo'] else args.method,rng)
        if args.method in ['grpo','starpo_s','gigpo']:
            from scripts.hard_metrics import group_counts
            diagnostic=branches if cfg['flat_group_size']==4 else [b for b in branches if b.lineage in ['0','1','2','3']]
            stats.update(group_counts(diagnostic,len(tasks)))
            stats['initial_group_source']='first four original trajectories before filtering'
        if cfg.get('response_protocol')=='first_complete_action_block':
            from response_protocol import protocol_metrics
            stats['response_protocol_metrics']=protocol_metrics(branches)
        sampled_time=time.monotonic()-t0
        atomic_json(out/'progress.json',dict(phase='optimization',method=args.method,step=step,
                      trajectories=len(branches),updated_unix=time.time()))
        optimizer_rng=np.random.default_rng(np.random.SeedSequence([seed,step,1702]))
        changes=update(actor,ref,optimizer,branches,cfg,args.method,tokenizer,optimizer_rng,len(tasks))
        sync_weights(actor,engine)
        elapsed=previous_elapsed+time.monotonic()-start
        record=dict(step=step,method=args.method,seed=seed,rollout_seconds=sampled_time,
                    resumed_from_step=start_step,
                    step_seconds=time.monotonic()-t0,elapsed_seconds=elapsed,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_allocated_by_gpu_gib=[torch.cuda.max_memory_allocated(i)/2**30 for i in range(torch.cuda.device_count())],
                    time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**stats,**changes)
        append_json(out/'train_metrics.jsonl',record)
        append_json(out/'branch_events.jsonl',dict(step=step,events=events))
        # Preserve complete raw sampled trajectories for auditing, including failures/pruning.
        rollout_path = out / f'rollouts_{step:04d}.jsonl.gz'
        if args.method == 'ucf':
            # The collector has persisted every actual trajectory once, before optimization.
            os.link(out / 'ucf_logs' / f'raw_leaves_{step:04d}.jsonl.gz', rollout_path)
        else:
            write_rollouts(rollout_path, branches, tasks)
        print(json.dumps(record),flush=True)
        atomic_json(out/'status.json',dict(status='training',method=args.method,step=step,total_steps=cfg['steps'],pid=os.getpid(),last=record))
        if step%cfg['save_every']==0 or step==cfg['steps']:
            save_checkpoint(cp,actor,optimizer,step,rng,cfg,elapsed)
        if step%cfg['eval_every']==0 or step==cfg['steps']:
            m=evaluate(engine,tokenizer,cfg,validation,out/f'validation_{step:04d}.json',100042,'validation')
            print(json.dumps({'validation_step':step,**m}),flush=True)
        if args.stop_after_step and step>=args.stop_after_step:
            break
    # A checkpoint at step 400 may still have final evaluation pending. Respect
    # the run window before starting that separate, full 100 x 4 evaluation.
    if step<cfg['steps'] or STOP:
        save_checkpoint(cp,actor,optimizer,step,rng,cfg,previous_elapsed+time.monotonic()-start)
        atomic_json(out/'status.json',dict(status='paused_checkpointed',step=step,total_steps=cfg['steps'],method=args.method,
                    final_evaluation_pending=step==cfg['steps']))
        return
    tests=data['validation'][4:8] if args.smoke else data['test']
    final=evaluate(engine,tokenizer,cfg,tests,out/'final_test.json',200042,'smoke_validation' if args.smoke else 'test')
    actor.save_pretrained(out/'final_model',safe_serialization=True)
    tokenizer.save_pretrained(out/'final_model')
    atomic_json(out/'status.json',dict(status='smoke_complete' if args.smoke else 'complete',step=step,total_steps=cfg['steps'],method=args.method,final=final))
    print(json.dumps({'FINAL':final}),flush=True)


if __name__=='__main__':
    main()
