"""Bootstrap vLLM for exported text-only Qwen3.5, then sync exact actor weights."""
import json
from pathlib import Path

_METADATA={'architectures','transformers_version','dtype','torch_dtype','use_cache','_name_or_path'}

def engine_source(model_path,base=None):
    source=Path(model_path)
    saved=json.loads((source/'config.json').read_text())
    if saved.get('model_type')!='qwen3_5_text':
        return source
    if base is None:
        raise ValueError('Point --model-dir at the original Qwen3.5 checkpoint with its full config')
    original=json.loads((Path(base)/'config.json').read_text())
    assert original.get('model_type')=='qwen3_5', 'Missing compatible Qwen3.5 bootstrap model'
    clean=lambda d:{k:v for k,v in d.items() if k not in _METADATA and v is not None}
    assert clean(saved)==clean(original['text_config']), 'Exported text architecture differs from bootstrap model'
    return Path(base)

def build_engine(actor,model_path,factory,synchronize,*,base=None,**kwargs):
    source=engine_source(model_path,base)
    engine=factory(model=str(source),tokenizer=str(model_path),**kwargs)
    if source!=Path(model_path):
        # Must finish before any validation, test or training collection is possible.
        synchronize(actor,engine)
    return engine
