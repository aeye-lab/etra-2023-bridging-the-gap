datasets = {
    'gazebase_all_sr1000_sl1000_dxy': {
        'repo_key': 'gazebase_all_sr1000_sl1000',
        'X_channels': ['dx', 'dy'],
        'label': 'subject_id',
    },
    'gazebase_all_sr1000_sl5000_dxy': {
        'repo_key': 'gazebase_all_sr1000_sl5000',
        'X_channels': ['dx', 'dy'],
        'label': 'subject_id',
    },
    'gazebase_all_sr1000_sl5000_savgol_maxvel500_dxy': {
        'repo_key': 'gazebase_all_sr1000_sl5000_savgol_maxvel500',
        'X_channels': ['dx', 'dy'],
        'label': 'subject_id',
    },
    'gazebase_all_sr1000_sl5000_savgol_maxvel750_dxy': {
        'repo_key': 'gazebase_all_sr1000_sl5000_savgol_maxvel750',
        'X_channels': ['dx', 'dy'],
        'label': 'subject_id',
    },
    'gazebase_all_sr1000_sl5000_savgol_maxvel1000_dxy': {
        'repo_key': 'gazebase_all_sr1000_sl5000_savgol_maxvel1000',
        'X_channels': ['dx', 'dy'],
        'label': 'subject_id',
    },
    'gazebase_all_sr1000_sl10000_dxy': {
        'repo_key': 'gazebase_all_sr1000_sl10000',
        'X_channels': ['dx', 'dy'],
        'label': 'subject_id',
    },
    'gazebase_task_all_sr1000_sl1000_dxy': {
        'repo_key': 'gazebase_task_all_sr1000_sl1000',
        'X_channels': ['dx', 'dy'],
        'label': 'task_id',
    },
    'gazebase_task_all_sr1000_sl5000_dxy': {
        'repo_key': 'gazebase_task_all_sr1000_sl5000',
        'X_channels': ['dx', 'dy'],
        'label': 'task_id',
    },
    'gazebase_task_all_sr1000_sl10000_dxy': {
        'repo_key': 'gazebase_task_all_sr1000_sl10000',
        'X_channels': ['dx', 'dy'],
        'label': 'task_id',
    },
    'gazebase_allnoblg_sr1000_sl1000_dxy': {
        'repo_key': 'gazebase_allnoblg_sr1000_sl1000',
        'X_channels': ['dx', 'dy'],
        'label': 'subject_id',
    },
    'gazebase_allnoblg_sr1000_sl5000_dxy': {
        'repo_key': 'gazebase_allnoblg_sr1000_sl5000',
        'X_channels': ['dx', 'dy'],
        'label': 'subject_id',
    },
    'judo_sr1000_sl1000_lx': {
        'repo_key': 'judo_sr1000_sl1000',
        'X_channels': ['dxl'],
        'label': 'subject_id',
    },
    'judo_sr1000_sl1000_ly': {
        'repo_key': 'judo_sr1000_sl1000',
        'X_channels': ['dyl'],
        'label': 'subject_id',
    },
    'judo_sr1000_sl1000_rx': {
        'repo_key': 'judo_sr1000_sl1000',
        'X_channels': ['dxr'],
        'label': 'subject_id',
    },
    'judo_sr1000_sl1000_ry': {
        'repo_key': 'judo_sr1000_sl1000',
        'X_channels': ['dyr'],
        'label': 'subject_id',
    },
    'judo_sr1000_sl1000_lxy': {
        'repo_key': 'judo_sr1000_sl1000',
        'X_channels': ['dxl', 'dyl'],
        'label': 'subject_id',
    },
    'judo_sr1000_sl1000_rxy': {
        'repo_key': 'judo_sr1000_sl1000',
        'X_channels': ['dxr', 'dyr'],
        'label': 'subject_id',
    },
    'judo_sr1000_sl1000_bxy': {
        'repo_key': 'judo_sr1000_sl1000',
        'X_channels': ['dxr', 'dyr', 'dxl', 'dyl'],
        'label': 'subject_id',
    },
    'judo_sr1000_sl1000_savgol_maxvel500_bxy': {
        'repo_key': 'judo_sr1000_sl1000_savgol_maxvel500',
        'X_channels': ['dxr', 'dyr', 'dxl', 'dyl'],
        'label': 'subject_id',
    },
    'judo_sr1000_sl1000_savgol_maxvel750_bxy': {
        'repo_key': 'judo_sr1000_sl1000_savgol_maxvel750',
        'X_channels': ['dxr', 'dyr', 'dxl', 'dyl'],
        'label': 'subject_id',
    },
    'judo_sr1000_sl1000_savgol_maxvel1000_bxy': {
        'repo_key': 'judo_sr1000_sl1000_savgol_maxvel1000',
        'X_channels': ['dxr', 'dyr', 'dxl', 'dyl'],
        'label': 'subject_id',
    },
    'judo_sr1000_sl1000_labs': {
        'repo_key': 'judo_sr1000_sl1000',
        'X_channels': ['d_abs_left'],
        'label': 'subject_id',
    },
    'judo_sr1000_sl1000_rabs': {
        'repo_key': 'judo_sr1000_sl1000',
        'X_channels': ['d_abs_right'],
        'label': 'subject_id',
    },
    'judo_sr1000_sl1000_babs': {
        'repo_key': 'judo_sr1000_sl1000',
        'X_channels': ['d_abs_left', 'd_abs_right'],
        'label': 'subject_id',
    },
    'mnist1d_sl1000': {
        'repo_key': 'mnist1d_sl1000',
        'X_channels': ['signal'],
        'label': 'class',
    },
    'mnist1d_sl1000_nonoise': {
        'repo_key': 'mnist1d_sl1000_nonoise',
        'X_channels': ['signal'],
        'label': 'class',
    },
    'potec_sr1000_sl1000_dx': {
        'repo_key': 'potec_sr1000_sl1000',
        'X_channels': ['dx'],
        'label': 'subject_id',
    },
    'potec_sr1000_sl1000_dy': {
        'repo_key': 'potec_sr1000_sl1000',
        'X_channels': ['dy'],
        'label': 'subject_id',
    },
    'potec_sr1000_sl1000_dxy': {
        'repo_key': 'potec_sr1000_sl1000',
        'X_channels': ['dx', 'dy'],
        'label': 'subject_id',
    },
    'potec_sr1000_sl1000_savgol_maxvel500_dxy': {
        'repo_key': 'potec_sr1000_sl1000_savgol_maxvel500',
        'X_channels': ['dx', 'dy'],
        'label': 'subject_id',
    },
    'potec_sr1000_sl1000_savgol_maxvel750_dxy': {
        'repo_key': 'potec_sr1000_sl1000_savgol_maxvel750',
        'X_channels': ['dx', 'dy'],
        'label': 'subject_id',
    },
    'potec_sr1000_sl1000_savgol_maxvel1000_dxy': {
        'repo_key': 'potec_sr1000_sl1000_savgol_maxvel1000',
        'X_channels': ['dx', 'dy'],
        'label': 'subject_id',
    },
    'potec_sr1000_sl1000_dabs': {
        'repo_key': 'potec_sr1000_sl1000',
        'X_channels': ['dabs'],
        'label': 'subject_id',
    },
}
