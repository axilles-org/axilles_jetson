"""Explicit dashboard selection: the ML models offered in the Launch menu.

The TBE fixed-profile controller is always listed (see server.list_models).
Adding an export to runs/ must not automatically expose an experimental
architecture or ablation in the hardware launch menu.
"""

MODEL_NAMES = {
    'model_film_subject_embedding_current_action_terrain':
        'model_film_subject_embedding_current_action_terrain',
    'deploy': 'model_subject_embedding_current_action',
    'model_obs_aug_subject_embedding_current_action':
        'model_obs_aug_subject_embedding_current_action',
}
