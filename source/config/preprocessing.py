# TODO: use this in Config class
pipelines = {
    'savgol_maxvel500': {
        'pos2vel_method': 'savitzky_golay',
        'pos2vel_kwargs': {
            'window_length': 7,
            'polyorder': 2,
            'deriv': 1,
            'delta': 1.0,
            'axis': -1,
            'mode': 'nearest',
        },
        'max_velocity': 500,
        'nan_threshold': 0.5,
    },
    'savgol_maxvel750': {
        'pos2vel_method': 'savitzky_golay',
        'pos2vel_kwargs': {
            'window_length': 7,
            'polyorder': 2,
            'deriv': 1,
            'delta': 1.0,
            'axis': -1,
            'mode': 'nearest',
        },
        'max_velocity': 750,
        'nan_threshold': 0.5,
    },
    'savgol_maxvel1000': {
        'pos2vel_method': 'savitzky_golay',
        'pos2vel_kwargs': {
            'window_length': 7,
            'polyorder': 2,
            'deriv': 1,
            'delta': 1.0,
            'axis': -1,
            'mode': 'nearest',
        },
        'max_velocity': 1000,
        'nan_threshold': 0.5,
    },
    'smooth_maxvel1000': {
        'pos2vel_method': 'smooth',
        'pos2vel_kwargs': {},
        'max_velocity': 1000,
        'nan_threshold': 0.5,
    },
    'smooth_maxvel500': {
        'pos2vel_method': 'smooth',
        'pos2vel_kwargs': {},
        'max_velocity': 500,
        'nan_threshold': 0.5,
    },
}
