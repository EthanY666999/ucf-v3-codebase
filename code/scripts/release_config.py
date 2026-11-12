"""Validate the UCF release before constructing a model or restoring state."""

def check_release(cfg, method=None):
    if (cfg.get('ucf_v3') or cfg.get('fork_point') == 'first_fence'
            or cfg.get('anchor16') or cfg.get('ucf_probability_source')):
        raise ValueError('UCF V3 checkpoint/config: use Git tag v3-final to restore it.')
    if method == 'ucf':
        if cfg.get('ucf_variant') != 'turn_resample':
            raise ValueError('Expected a turn_resample config. For UCF V3 use Git tag v3-final.')
        if cfg['ucf_estimator'] not in ('unbiased', 'merge', 'success'):
            raise ValueError('Unknown UCF estimator')
        if type(cfg['ucf_resamples']) is not int or cfg['ucf_resamples'] < 1:
            raise ValueError('ucf_resamples must be a positive integer')
        if cfg['ucf_turn_prior'] not in ('uniform', 'inverse_turn'):
            raise ValueError('Unknown UCF turn prior')
