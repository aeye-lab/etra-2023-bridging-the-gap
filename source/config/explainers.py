from captum import attr as captum_attr
import zennit.composites

from explainers.zennit import EpsilonAlphaBeta
from explainers.zennit import EpsilonAlphaBetaFlat
from explainers.zennit import EpsilonOnly
from explainers.zennit import EpsilonOnlyDense
from explainers.zennit import EpsilonOnlyConv
from explainers.zennit import EpsilonFlat
from explainers.zennit import EpsilonZPlusFlat
from explainers.zennit import EpsilonGamma
from explainers.zennit import EpsilonGammaFlat


explainers = {
    'deeplift_zero': {
        'name': 'DeepLift (zero baseline)',
        'class': captum_attr.DeepLift,
        'framework': 'captum',
        'supported_model_frameworks': ['pytorch'],
    },
    'deeplift_mean': {
        'name': 'DeepLift (mean baseline)',
        'class': captum_attr.DeepLift,
        'baseline': 'mean',
        'framework': 'captum',
        'supported_model_frameworks': ['pytorch'],
    },
    'deeplift_black': {
        'name': 'DeepLift (extreme negative baseline)',
        'class': captum_attr.DeepLift,
        'baseline': 'extreme_negative',
        'framework': 'captum',
        'supported_model_frameworks': ['pytorch'],
    },
    'deeplift_white': {
        'name': 'DeepLift (extreme positive baseline)',
        'class': captum_attr.DeepLift,
        'baseline': 'extreme_positive',
        'framework': 'captum',
        'supported_model_frameworks': ['pytorch'],
    },
    'deeplift_mean_smoothgrad_n50_std0.1': {
        'name': 'DeepLift (mean baseline) w/ smoothgrad (n=50, std=0.1)',
        'class': captum_attr.DeepLift,
        'attribute_kwargs': {
            'nt_type': 'smoothgrad',
            'nt_samples': 50,
            'stdevs': 0.1,
        },
        'baseline': 'mean',
        'framework': 'captum',
        'supported_model_frameworks': ['pytorch'],
    },
    'deeplift_mean_smoothgrad_n50_std0.2': {
        'name': 'DeepLift (mean baseline) w/ smoothgrad (n=50, std=0.2)',
        'class': captum_attr.DeepLift,
        'attribute_kwargs': {
            'nt_type': 'smoothgrad',
            'nt_samples': 50,
            'stdevs': 0.2,
        },
        'baseline': 'mean',
        'framework': 'captum',
        'supported_model_frameworks': ['pytorch'],
    },
    'deeplift_mean_smoothgrad_n50_std0.3': {
        'name': 'DeepLift (mean baseline) w/ smoothgrad (n=50, std=0.3)',
        'class': captum_attr.DeepLift,
        'attribute_kwargs': {
            'nt_type': 'smoothgrad',
            'nt_samples': 50,
            'stdevs': 0.3,
        },
        'baseline': 'mean',
        'framework': 'captum',
        'supported_model_frameworks': ['pytorch'],
    },
    'deeplift_mean_smoothgrad_n50_std0.4': {
        'name': 'DeepLift (mean baseline) w/ smoothgrad (n=50, std=0.4)',
        'class': captum_attr.DeepLift,
        'attribute_kwargs': {
            'nt_type': 'smoothgrad',
            'nt_samples': 50,
            'stdevs': 0.4,
        },
        'baseline': 'mean',
        'framework': 'captum',
        'supported_model_frameworks': ['pytorch'],
    },
    'deeplift_mean_smoothgrad_n50_std0.5': {
        'name': 'DeepLift (mean baseline) w/ smoothgrad (n=50, std=0.5)',
        'class': captum_attr.DeepLift,
        'attribute_kwargs': {
            'nt_type': 'smoothgrad',
            'nt_samples': 50,
            'stdevs': 0.5,
        },
        'baseline': 'mean',
        'framework': 'captum',
        'supported_model_frameworks': ['pytorch'],
    },
    'deconvolution': {
        'name': 'Devonvolution',
        'class': captum_attr.Deconvolution,
        'framework': 'captum',
        'supported_model_frameworks': ['pytorch'],
    },
    'feature_permutation': {
        'name': 'Feature Permutation',
        'class': captum_attr.FeaturePermutation,
        'framework': 'captum',
        'supported_model_frameworks': ['pytorch'],
    },
    'guided_backprop': {
        'name': 'Guided Backpropagation',
        'class': captum_attr.GuidedBackprop,
        'framework': 'captum',
        'supported_model_frameworks': ['pytorch'],
    },
    'integrated_gradients_zero': {
        'name': 'Integrated Gradients (zero baseline)',
        'class': captum_attr.IntegratedGradients,
        'framework': 'captum',
        'supported_model_frameworks': ['pytorch'],
        'attribute_kwargs': {
            'internal_batch_size': 64,
        },
    },
    'integrated_gradients_mean': {
        'name': 'Integrated Gradients (mean baseline)',
        'class': captum_attr.IntegratedGradients,
        'baseline': 'mean',
        'framework': 'captum',
        'supported_model_frameworks': ['pytorch'],
        'attribute_kwargs': {
            'internal_batch_size': 64,
        },
    },
    'integrated_gradients_black': {
        'name': 'Integrated Gradients (extreme negative baseline)',
        'class': captum_attr.IntegratedGradients,
        'baseline': 'extreme_negative',
        'framework': 'captum',
        'supported_model_frameworks': ['pytorch'],
        'attribute_kwargs': {
            'internal_batch_size': 64,
        },
    },
    'integrated_gradients_white': {
        'name': 'Integrated Gradients (extreme positive baseline)',
        'class': captum_attr.IntegratedGradients,
        'baseline': 'extreme_positive',
        'framework': 'captum',
        'supported_model_frameworks': ['pytorch'],
        'attribute_kwargs': {
            'internal_batch_size': 64,
        }
    },
    'integrated_gradients_mean_smoothgrad_n50_std0.1': {
        'name': 'Integrated Gradients (mean baseline) w/ smoothgrad (n=50, std=0.1)',
        'class': captum_attr.IntegratedGradients,
        'attribute_kwargs': {
            'nt_type': 'smoothgrad',
            'nt_samples': 50,
            'stdevs': 0.1,
            'internal_batch_size': 64,
        },
        'baseline': 'mean',
        'framework': 'captum',
        'supported_model_frameworks': ['pytorch'],
    },
    'integrated_gradients_mean_smoothgrad_n50_std0.2': {
        'name': 'Integrated Gradients (mean baseline) w/ smoothgrad (n=50, std=0.2)',
        'class': captum_attr.IntegratedGradients,
        'attribute_kwargs': {
            'nt_type': 'smoothgrad',
            'nt_samples': 50,
            'stdevs': 0.2,
            'internal_batch_size': 64,
        },
        'baseline': 'mean',
        'framework': 'captum',
        'supported_model_frameworks': ['pytorch'],
    },
    'integrated_gradients_mean_smoothgrad_n50_std0.3': {
        'name': 'Integrated Gradients (mean baseline) w/ smoothgrad (n=50, std=0.3)',
        'class': captum_attr.IntegratedGradients,
        'attribute_kwargs': {
            'nt_type': 'smoothgrad',
            'nt_samples': 50,
            'stdevs': 0.3,
            'internal_batch_size': 64,
        },
        'baseline': 'mean',
        'framework': 'captum',
        'supported_model_frameworks': ['pytorch'],
    },
    'integrated_gradients_mean_smoothgrad_n50_std0.4': {
        'name': 'Integrated Gradients (mean baseline) w/ smoothgrad (n=50, std=0.4)',
        'class': captum_attr.IntegratedGradients,
        'attribute_kwargs': {
            'nt_type': 'smoothgrad',
            'nt_samples': 50,
            'stdevs': 0.4,
            'internal_batch_size': 64,
        },
        'baseline': 'mean',
        'framework': 'captum',
        'supported_model_frameworks': ['pytorch'],
    },
    'integrated_gradients_mean_smoothgrad_n50_std0.5': {
        'name': 'Integrated Gradients (mean baseline) w/ smoothgrad (n=50, std=0.5)',
        'class': captum_attr.IntegratedGradients,
        'attribute_kwargs': {
            'nt_type': 'smoothgrad',
            'nt_samples': 50,
            'stdevs': 0.5,
            'internal_batch_size': 64,
        },
        'baseline': 'mean',
        'framework': 'captum',
        'supported_model_frameworks': ['pytorch'],
    },
    'input_x_gradient': {
        'name': 'Input X Gradient',
        'class': captum_attr.InputXGradient,
        'framework': 'captum',
        'supported_model_frameworks': ['pytorch'],
    },
    'occlusion_s3': {
        'name': 'Occlusion (patch size = 3)',
        'class': captum_attr.Occlusion,
        'attribute_kwargs': {
            'sliding_window_shapes': (1, 3),
            'perturbations_per_eval': 125,
        },
        'baselines': 0,
        'framework': 'captum',
        'supported_model_frameworks': ['pytorch'],
    },
    'occlusion_s5': {
        'name': 'Occlusion (patch size = 5)',
        'class': captum_attr.Occlusion,
        'attribute_kwargs': {
            'sliding_window_shapes': (1, 5),
            'perturbations_per_eval': 125,
        },
        'baselines': 0,
        'framework': 'captum',
        'supported_model_frameworks': ['pytorch'],
    },
    'saliency': {
        'name': 'Saliency',
        'class': captum_attr.Saliency,
        'attribute_kwargs': {
            'abs': False,
        },
        'framework': 'captum',
        'supported_model_frameworks': ['pytorch'],
    },

    'lrp_epsilon_plus': {
        'name': 'LRP-ε+',
        'class': zennit.composites.EpsilonPlus,
        'init_kwargs': {'epsilon': 1e-6},
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },
    'lrp_epsilon0_plus': {
        'name': 'LRP-ε+ (ε = 0)',
        'class': zennit.composites.EpsilonPlus,
        'init_kwargs': {'epsilon': 0.0},
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },
    'lrp_epsilon1e-10_plus': {
        'name': 'LRP-ε+ (ε = 1e-10)',
        'class': zennit.composites.EpsilonPlus,
        'init_kwargs': {'epsilon': 1e-10},
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },
    'lrp_epsilon1e-8_plus': {
        'name': 'LRP-ε+ (ε = 1e-8)',
        'class': zennit.composites.EpsilonPlus,
        'init_kwargs': {'epsilon': 1e-8},
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },
    'lrp_epsilon1e-6_plus': {
        'name': 'LRP-ε+ (ε = 1e-6)',
        'class': zennit.composites.EpsilonPlus,
        'init_kwargs': {'epsilon': 1e-6},
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },
    'lrp_epsilon1e-4_plus': {
        'name': 'LRP-ε+ (ε = 1e-4)',
        'class': zennit.composites.EpsilonPlus,
        'init_kwargs': {'epsilon': 1e-4},
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },
    'lrp_epsilon1e-2_plus': {
        'name': 'LRP-ε+ (ε = 1e-2)',
        'class': zennit.composites.EpsilonPlus,
        'init_kwargs': {'epsilon': 1e-2},
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },


    'lrp_epsilon_plus_flat': {
        'name': 'LRP-ε+♭',
        'class': zennit.composites.EpsilonPlusFlat,
        'init_kwargs': {'epsilon': 1e-6},
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },
    'lrp_epsilon0_plus_flat1': {
        'name': 'LRP-ε+♭ (ε = 0)',
        'class': zennit.composites.EpsilonPlusFlat,
        'init_kwargs': {'epsilon': 0.0},
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },
    'lrp_epsilon1e-10_plus_flat1': {
        'name': 'LRP-ε+♭ (ε = 1e-10)',
        'class': zennit.composites.EpsilonPlusFlat,
        'init_kwargs': {'epsilon': 1e-10},
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },
    'lrp_epsilon1e-8_plus_flat1': {
        'name': 'LRP-ε+♭ (ε = 1e-8)',
        'class': zennit.composites.EpsilonPlusFlat,
        'init_kwargs': {'epsilon': 1e-8},
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },
    'lrp_epsilon1e-6_plus_flat1': {
        'name': 'LRP-ε+♭ (ε = 1e-6)',
        'class': zennit.composites.EpsilonPlusFlat,
        'init_kwargs': {'epsilon': 1e-6},
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },
    'lrp_epsilon1e-4_plus_flat1': {
        'name': 'LRP-ε+♭ (ε = 1e-4)',
        'class': zennit.composites.EpsilonPlusFlat,
        'init_kwargs': {'epsilon': 1e-4},
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },
    'lrp_epsilon1e-2_plus_flat1': {
        'name': 'LRP-ε+♭ (ε = 1e-2)',
        'class': zennit.composites.EpsilonPlusFlat,
        'init_kwargs': {'epsilon': 1e-2},
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },



    'lrp_epsilon0_only': {
        'name': 'LRP-ε (ε = 0)',
        'class': EpsilonOnly,
        'init_kwargs': {'epsilon': 0},
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },
    'lrp_epsilon0.25_only': {
        'name': 'LRP-ε (ε = 0.25)',
        'class': EpsilonOnly,
        'init_kwargs': {'epsilon': 0.25},
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },
    'lrp_epsilon0.1_only': {
        'name': 'LRP-ε (ε = 0.1)',
        'class': EpsilonOnly,
        'init_kwargs': {'epsilon': 0.1},
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },
    'lrp_epsilon0.01_only': {
        'name': 'LRP-ε (ε = 0.01)',
        'class': EpsilonOnly,
        'init_kwargs': {'epsilon': 0.01},
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },
    'lrp_epsilon1e-6_only': {
        'name': 'LRP-ε (ε = 1e-6)',
        'class': EpsilonOnly,
        'init_kwargs': {'epsilon': 1e-6},
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },

    'lrp_epsilon_only_dense': {
        'name': 'LRP-ε',
        'class': EpsilonOnlyDense,
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },
    'lrp_epsilon_only_conv': {
        'name': 'LRP-ε (dense+conv)',
        'class': EpsilonOnlyConv,
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },
    'lrp_epsilon_flat': {
        'name': 'LRP-εb',
        'class': EpsilonFlat,
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },
    'lrp_epsilon_zplus': {
        'name': 'LRP-εz+',
        'class': zennit.composites.EpsilonPlus,
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },
    'lrp_epsilon_zplus_flat': {
        'name': 'LRP-εz+b',
        'class': EpsilonZPlusFlat,
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },

    'lrp_epsilon_gamma0.25': {
        'name': 'LRP-εγ',
        'class': EpsilonGamma,
        'init_kwargs': {'epsilon': 1e-6, 'gamma': 0.25},
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },
    'lrp_epsilon_gamma0.1': {
        'name': 'LRP-εγ',
        'class': EpsilonGamma,
        'init_kwargs': {'epsilon': 1e-6, 'gamma': 0.1},
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },
    'lrp_epsilon_gamma0.25_flat': {
        'name': 'LRP-εγb',
        'class': EpsilonGammaFlat,
        'init_kwargs': {'epsilon': 1e-6, 'gamma': 0.25},
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },
    'lrp_epsilon_gamma0.1_flat': {
        'name': 'LRP-εγb',
        'class': EpsilonGammaFlat,
        'init_kwargs': {'epsilon': 1e-6, 'gamma': 0.1},
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },


    'lrp_epsilon_alpha1_beta0': {
        'name': 'LRP-εγb',
        'class': EpsilonAlphaBeta,
        'init_kwargs': {'epsilon': 1e-6, 'alpha': 1, 'beta': 0},
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },
    'lrp_epsilon_alpha2_beta1': {
        'name': 'LRP-εγb',
        'class': EpsilonAlphaBeta,
        'init_kwargs': {'epsilon': 1e-6, 'alpha': 2, 'beta': 1},
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },
    'lrp_epsilon_alpha1_beta0_flat': {
        'name': 'LRP-εγb',
        'class': EpsilonAlphaBetaFlat,
        'init_kwargs': {'epsilon': 1e-6, 'alpha': 1, 'beta': 0},
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },
    'lrp_epsilon_alpha2_beta1_flat': {
        'name': 'LRP-εγb',
        'class': EpsilonAlphaBetaFlat,
        'init_kwargs': {'epsilon': 1e-6, 'alpha': 2, 'beta': 1},
        'framework': 'zennit',
        'supported_model_frameworks': ['pytorch'],
    },

    'lrp_captum_epsilon': {
        'name': 'LRP-ε (captum)',
        'class': captum_attr.LRP,
        'framework': 'captum',
        'supported_model_frameworks': ['pytorch'],
    },

    '/': {},
}
