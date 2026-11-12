"""Expose the current actor to UCF probability calculation; sampler stays frozen."""
def configure_builder(train_module):
 if getattr(train_module.build_models,'_ucf_runtime',False):return
 original=train_module.build_models
 def build_models(cfg,seed):
  actor,ref,engine,tokenizer=original(cfg,seed)
  engine._ucf_actor=actor
  return actor,ref,engine,tokenizer
 build_models._ucf_runtime=True
 train_module.build_models=build_models
