"""Explicit dashboard selection: subject-embedding TCN variants and subject FiLM.

Adding an export to runs/ must not automatically expose an experimental
architecture or ablation in the hardware launch menu.
"""

MODEL_NAMES = {
    'model_film_subject_embedding_current_action_terrain':
        'model_film_subject_embedding_current_action_terrain',
    'deploy': 'model_subject_embedding_current_action',
    **{f'model_subject_embedding_future_action_{h}':
       f'model_subject_embedding_future_action_{h}' for h in (50, 100, 150)},
    'model_obs_aug_subject_embedding_current_action':
        'model_obs_aug_subject_embedding_current_action',
    **{f'model_obs_aug_subject_embedding_future_action_{h}':
       f'model_obs_aug_subject_embedding_future_action_{h}' for h in (50, 100, 150)},
    'model_obs_aug_subject_embedding_no_gyro_current_action':
        'model_obs_aug_subject_embedding_no_gyro_current_action',
    **{f'model_obs_aug_subject_embedding_no_gyro_future_action_{h}':
       f'model_obs_aug_subject_embedding_no_gyro_future_action_{h}' for h in (50, 100, 150)},
    'model_obs_aug_subject_embedding_no_imu_current_action':
        'model_obs_aug_subject_embedding_no_imu_current_action',
}
