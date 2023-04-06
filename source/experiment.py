from typing import List, Optional, Union


class ExperimentKeys:

    def __init__(
        self,
        data_key: str,
        model_key: Optional[str] = None,
        explainer_key: Optional[str] = None,
        metric_key: Optional[str] = None,
        segmentation_key: Optional[str] = None,
        n_augmentations: Optional[int] = 0,
        noise_std: Optional[float] = 0.0,
    ):
        self.data = self.data_key_map(data_key)
        self.model = model_key
        self.explainer = explainer_key
        self.metric = metric_key
        self.segmentation = segmentation_key

        # TODO: create augmentation/trainer key
        self.n_augmentations = n_augmentations
        self.noise_std = noise_std

    @classmethod
    def data_key_map(cls, data_key: str):
        if data_key in ['mnist1d', 'mnist1d-1000']:
            data_key = 'mnist1d_sl1000'

        elif data_key == 'gazebase-1000':
            data_key = 'gazebase_all_sr1000_sl1000_savgol_maxvel1000_dxy'

        elif data_key in ['gazebase', 'gazebase-5000']:
            data_key = 'gazebase_all_sr1000_sl5000_savgol_maxvel1000_dxy'

        elif data_key in ['judo', 'judo-bino', 'judo-1000-bino']:
            data_key = 'judo_sr1000_sl1000_savgol_maxvel1000_bxy'

        elif data_key in ['judo-mono', 'judo-1000-mono']:
            data_key = 'judo_sr1000_sl1000_rxy'

        elif data_key in ['potec', 'potec-1000']:
            data_key = 'potec_sr1000_sl1000_savgol_maxvel1000_dxy'

        return data_key

    def __repr__(self):
        return (
            'ExperimentKeys('
            f'data={self.data}'
            f', model={self.model}'
            f', explainer={self.explainer}'
            f', metric={self.metric}'
            f', segmentation={self.segmentation}'
            ')'
        )


class ExperimentKeyFactory:

    def __init__(
        self,
        data_key: Union[str, List[str]],
        model_key: Union[str, List[str]],
        explainer_key: Optional[Union[str, List[str]]] = None,
        metric_key: Optional[Union[str, List[str]]] = None,
        segmentation_key: Optional[Union[str, List[str]]] = None,
        n_augmentations: Optional[int] = 0,
        noise_std: Optional[float] = 0.0,
    ):
        self.data = self.expand_data_key(data_key)
        self.model = self.expand_model_key(model_key)
        self.explainer = self.expand_explainer_key(explainer_key)
        self.metric = self.expand_metric_key(metric_key)
        self.segmentation = self.expand_segmentation_key(segmentation_key)

        self.n_augmentations = n_augmentations
        self.noise_std = noise_std

    def expand_data_key(self, data_key: Union[List[str], str]):
        if data_key == 'all':
            data_keys = [
                'gazebase-1000',
                'gazebase-5000',
                'judo-bino',
                'judo-mono',
                'potec',
            ]

        elif data_key == 'gazebase-clip-thresholds':
            data_keys = [
                'gazebase_all_sr1000_sl5000_savgol_maxvel500_dxy',
                'gazebase_all_sr1000_sl5000_savgol_maxvel750_dxy',
                'gazebase_all_sr1000_sl5000_savgol_maxvel1000_dxy',
            ]

        elif data_key == 'judo-clip-thresholds':
            data_keys = [
                'judo_sr1000_sl1000_savgol_maxvel500_bxy',
                'judo_sr1000_sl1000_savgol_maxvel750_bxy',
                'judo_sr1000_sl1000_savgol_maxvel1000_bxy',
            ]

        elif data_key == 'potec-clip-thresholds':
            data_keys = [
                'potec_sr1000_sl1000_savgol_maxvel500_dxy',
                'potec_sr1000_sl1000_savgol_maxvel750_dxy',
                'potec_sr1000_sl1000_savgol_maxvel1000_dxy',
            ]

        elif isinstance(data_key, str):
            data_keys = [data_key]
        elif isinstance(data_key, list):
            data_keys = data_key

        data_keys = [
            ExperimentKeys.data_key_map(data_key)
            for data_key in data_keys
        ]

        return data_keys

    def expand_model_key(self, model_key: Union[List[str], str]):
        if model_key == 'all':
            model_keys = [
                'eky2',
                'del_fast',
                'del_slow',
                'del',
            ]

        elif model_key == 'all-really':
            model_keys = [
                'eky2',

                'del_fast',
                'del_slow',
                'del',
                'del_singlestage',

                'del_fast_zstd',
                'del_slow_zstd',
                'del_zstd',
                'del_singlestage_zstd',
            ]

        elif model_key == 'del-complete':
            model_keys = [
                'del_fast',
                'del_slow',
                'del',
            ]

        elif model_key == 'del-complete-zstd':
            model_keys = [
                'del_fast_zstd',
                'del_slow_zstd',
                'del_zstd',
            ]

        elif isinstance(model_key, str):
            model_keys = [model_key]
        elif isinstance(model_key, list):
            model_keys = model_key

        return model_keys

    def expand_explainer_key(self, explainer_key: Union[List[str], str]):
        if explainer_key == None:
            explainer_keys = []

        elif explainer_key == 'etra23':
            explainer_keys = [
                'deeplift_zero',
                'integrated_gradients_zero',
                'lrp_epsilon0.25_only',
            ]

        elif explainer_key == 'captum':
            explainer_keys = [
                'input_x_gradient',
                'deeplift_zero',
                'integrated_gradients_zero',
                'occlusion_s3',
                'occlusion_s5',
            ]
        elif explainer_key == 'occlusion':
            explainer_keys = [
                'occlusion_s3',
                'occlusion_s5',
            ]
        elif explainer_key == 'gradient-captum':
            explainer_keys = [
                'input_x_gradient',
                'deeplift_zero',
                'integrated_gradients_zero',
            ]
        elif explainer_key == 'lrp':
            explainer_keys = [
                'lrp_epsilon0.25_only',

                'lrp_epsilon_plus',
                #'lrp_epsilon_plus_flat',

                'lrp_epsilon_gamma0.25',
                #'lrp_epsilon_gamma0.25_flat',
                #'lrp_epsilon_gamma0.1': 'LRP-εγ.1',
                #'lrp_epsilon_gamma0.1_flat': 'LRP-εγ.1♭',

                #'lrp_epsilon_alpha2_beta1',
                #'lrp_epsilon_alpha2_beta1_flat',
            ]
        elif explainer_key == 'smoothgrad':
            explainer_keys = [
                'deeplift_mean_smoothgrad_n50_std0.1',
                'deeplift_mean_smoothgrad_n50_std0.2',
                'deeplift_mean_smoothgrad_n50_std0.3',
                'deeplift_mean_smoothgrad_n50_std0.4',
                'deeplift_mean_smoothgrad_n50_std0.5',

                'integrated_gradients_mean_smoothgrad_n50_std0.1',
                'integrated_gradients_mean_smoothgrad_n50_std0.2',
                'integrated_gradients_mean_smoothgrad_n50_std0.3',
                'integrated_gradients_mean_smoothgrad_n50_std0.4',
                'integrated_gradients_mean_smoothgrad_n50_std0.5',
            ]
        elif isinstance(explainer_key, str):
            explainer_keys = [explainer_key]
        elif isinstance(explainer_key, list):
            explainer_keys = explainer_key

        return explainer_keys

    def expand_metric_key(self, metric_key: Union[List[str], str]):
        if metric_key == None:
            metric_keys = []

        elif metric_key == 'localization':
            metric_keys = [
                'area_under_curve',
                'attribution_localisation',
                'attribution_localisation_weighted',
                'localized_attribution_aggregate_max',
                'localized_attribution_aggregate_mean',
                'localized_attribution_aggregate_median',
                'pointing_game',
                'pointing_game_weighted',
                'relevance_rank_accuracy',
                'relevance_mass_accuracy',
                'top_1_percent_intersection',
                'top_1_percent_intersection_concept_influence',
                'top_2_percent_intersection',
                'top_2_percent_intersection_concept_influence',
                'top_5_percent_intersection',
                'top_5_percent_intersection_concept_influence',
                'top_10_percent_intersection',
                'top_10_percent_intersection_concept_influence',
                'top_20_percent_intersection',
                'top_20_percent_intersection_concept_influence',

            ]

        elif metric_key == 'etra23':
            metric_keys = [
                'top_2_percent_intersection',
                'top_2_percent_intersection_concept_influence',
            ]

        elif metric_key == 'region_perturbation':
            metric_keys = [
                'region_perturbation_s3_uniform_abs_nonorm_morf',
                'region_perturbation_s3_uniform_noabs_nonorm_morf',
                'region_perturbation_s3_uniform_abs_nonorm_lerf',
                'region_perturbation_s3_uniform_noabs_nonorm_lerf',

                'region_perturbation_s5_uniform_abs_nonorm_morf',
                'region_perturbation_s5_uniform_noabs_nonorm_morf',
                'region_perturbation_s5_uniform_abs_nonorm_lerf',
                'region_perturbation_s5_uniform_noabs_nonorm_lerf',
            ]

        elif metric_key == 'region_perturbation_random':
            metric_keys = [
                'region_perturbation_s3_uniform_noabs_nonorm_random',
                'region_perturbation_s5_uniform_noabs_nonorm_random',
            ]

        elif metric_key == 'region_perturbation_noabs_morf':
            metric_keys = [
                #'region_perturbation_s3_uniform_abs_nonorm_morf',
                'region_perturbation_s3_uniform_noabs_nonorm_morf',
                #'region_perturbation_s5_uniform_abs_nonorm_morf',
                'region_perturbation_s5_uniform_noabs_nonorm_morf',
            ]

        elif metric_key == 'region_perturbation_lerf':
            metric_keys = [
                'region_perturbation_s3_uniform_abs_nonorm_lerf',
                'region_perturbation_s3_uniform_noabs_nonorm_lerf',
                'region_perturbation_s5_uniform_abs_nonorm_lerf',
                'region_perturbation_s5_uniform_noabs_nonorm_lerf',
            ]

        elif metric_key == 'region_perturbation_s3':
            metric_keys = [
                'region_perturbation_s3_uniform_abs_nonorm_morf',
                'region_perturbation_s3_uniform_noabs_nonorm_morf',
                'region_perturbation_s3_uniform_abs_nonorm_lerf',
                'region_perturbation_s3_uniform_noabs_nonorm_lerf',
            ]

        elif metric_key == 'region_perturbation_s5':
            metric_keys = [
                'region_perturbation_s5_uniform_abs_nonorm_morf',
                'region_perturbation_s5_uniform_noabs_nonorm_morf',
                'region_perturbation_s5_uniform_abs_nonorm_lerf',
                'region_perturbation_s5_uniform_noabs_nonorm_lerf',
            ]
        elif metric_key == 'axiomatic-metrics':
            metric_keys = [
                'completeness',
                'nonsensitivity',
                'input_invariance',
            ]
        elif metric_key == 'complexity-metrics':
            metric_keys = [
                'complexity',
                'sparseness',
                'effective_complexity',
            ]
        elif metric_key == 'randomisation-metrics':
            metric_keys = [
                'model_parameter_randomisation',
                'random_logit',
            ]
        elif metric_key == 'robustness-metrics':
            metric_keys = [
                'local_lipschitz_estimate',
                'max_sensitivity',
                'avg_sensitivity',
                'continuity',
            ]
        elif metric_key == 'fast-metrics':
            metric_keys = [
                #'completeness',
                'sparseness',
                'complexity',
                #'effective_complexity',
            ]

        elif metric_key == 'all':
            metric_keys = [
                'completeness',
                'nonsensitivity',
                'input_invariance',

                'complexity',
                'sparseness',
                'effective_complexity',

                'model_parameter_randomisation',
                'random_logit',

                'local_lipschitz_estimate',
                'max_sensitivity',
                'avg_sensitivity',
                'continuity',
            ]

        elif isinstance(metric_key, str):
            metric_keys = [metric_key]
        elif isinstance(metric_key, list):
            metric_keys = metric_key

        return metric_keys

    @staticmethod
    def expand_segmentation_key(segmentation_key: Union[List[str], str]):
        if segmentation_key is None:
            segmentation_keys = []

        elif segmentation_key == 'all':
            segmentation_keys = [
                'nan',
                'clip',
                'unclassified',
                'ivt.fixation',
                'engbert.saccade',
            ]

        elif segmentation_key == 'etra23':
            segmentation_keys = [
                'nan',
                'clip',
                'unclassified',
                'ivt.fixation',
                'engbert.saccade',
            ]

        elif segmentation_key == 'etra23-bino':
            segmentation_keys = [
                'nan',
                'clip',
                'engbert.fixation',
                'engbert.saccade',
                'engbert.saccade.monocular',
            ]

        elif isinstance(segmentation_key, str):
            segmentation_keys = [segmentation_key]
        elif isinstance(segmentation_key, list):
            segmentation_keys = segmentation_key

        return segmentation_keys

    def __str__(self) -> str:
        return (
            f'dataset key: {self.data}\n'
            f'model key: {self.model}\n'
            f'explainer key: {self.explainer}\n'
            f'metric key: {self.metric}\n'
            f'segmentation key: {self.segmentation}\n'
            f'augmentation: {self.n_augmentations}, {self.noise_std}'
        )

    def print(self) -> None:
        print("###############################################################")
        print("###############################################################")
        print("Experiment Keys:")
        print(str(self))
        print("###############################################################")
        print("###############################################################")

    def iterate(self):
        for data_key in self.data:
            if not self.model:
                yield ExperimentKeys(
                    data_key=data_key,
                )
                continue

            for model_key in self.model:
                if not self.explainer:
                    yield ExperimentKeys(
                        data_key=data_key,
                        model_key=model_key,
                    )
                    continue

                for explainer_key in self.explainer:
                    if not self.metric:
                        yield ExperimentKeys(
                            data_key=data_key,
                            model_key=model_key,
                            explainer_key=explainer_key,
                        )
                        continue

                    for metric_key in self.metric:
                        if not self.segmentation:
                            yield ExperimentKeys(
                                data_key=data_key,
                                model_key=model_key,
                                explainer_key=explainer_key,
                                metric_key=metric_key,
                            )

                        for segmentation_key in self.segmentation:
                            yield ExperimentKeys(
                                data_key=data_key,
                                model_key=model_key,
                                explainer_key=explainer_key,
                                metric_key=metric_key,
                                segmentation_key=segmentation_key,
                            )
