import argparse
import re
from pathlib import Path
from typing import Dict, List

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quantus
import shutil
import torch
from pymovements.transforms import vnorm
from tqdm import tqdm

from config import Config
from datasets.loader import DataFold
from evaluate import Evaluator
from experiment import ExperimentKeyFactory, ExperimentKeys
from paths import ExperimentPaths, makedirs


segmentation_binning_regex = re.compile(
    r'(?P<base_segmentation_key>.+)'
    r'_binning'
    r'_(?P<binning_property>.+)'
    r'_n(?P<n_bins>\d+)'
)

segmentation_dissection_regex = re.compile(
    r'(?P<base_segmentation_key>.+)'
    r'_dissection'
    r'_(?P<dissection_method>.+)'
)


def do_property_binning(segmentation, df, column, n_bins=100, minimum=None, show_progress=True):
    if minimum is None:
        bins = np.linspace(df[column].min(), df[column].max(), num=n_bins)
    else:
        bins = np.linspace(minimum, df[column].max(), num=n_bins)

    binning = np.digitize(df[column], bins)

    binseg = np.zeros((segmentation.shape[0], n_bins, segmentation.shape[1]), dtype=bool)

    for (_, event_property), bin_id in tqdm(zip(df.iterrows(), binning), leave=False,
                                            disable=not show_progress,
                                            total=len(binning), desc='binning'):
        instance_id = int(event_property.instance_id)
        onset = int(event_property.onset)
        offset = int(event_property.offset + 1)

        binseg[instance_id, bin_id - 1, onset:offset] = True

    return bins, binseg


def do_event_dissection(velocities, segmentation, df, show_progress=True):
    n_sections = 6  # pre, rise, peak, basin, fall, post

    sequence_length = segmentation.shape[1]
    dissection = np.zeros((segmentation.shape[0], n_sections, sequence_length), dtype=bool)

    v_norm = vnorm(velocities, axis=1)

    for row_idx, event_property in tqdm(df.iterrows(), desc='dissecting', total=len(df),
                                        disable=not show_progress,
                                        leave=False):
        instance_id = int(event_property.instance_id)
        onset = max(0, int(event_property.onset))
        offset = min(sequence_length, int(event_property.offset + 1))
        duration = offset - onset - 1

        # initialize dissection masks
        event_mask = np.zeros(sequence_length, dtype=bool)
        event_mask[onset:offset] = True

        # set peak mask by intersecting >80% v_peak and event mask
        try:
            v_norm_instance = v_norm[instance_id]
        except:
            breakpoint()
        v_peak = v_norm_instance[onset:offset].max()
        peak_mask = np.logical_and(event_mask, v_norm_instance >= 0.8 * v_peak)

        # set rise/fall mask by getting peak mask onset and offset
        peak_onset = np.where(peak_mask)[0][0]
        peak_offset = np.where(peak_mask)[0][-1]
        rise_mask = np.zeros(sequence_length, dtype=bool)
        fall_mask = np.zeros(sequence_length, dtype=bool)
        rise_mask[onset:peak_onset] = True
        fall_mask[peak_offset+1:offset] = True

        # set basin mask for samples which are between peak onset and offset but <80% v_peak
        basin_mask = np.zeros(sequence_length, dtype=bool)
        basin_mask[peak_onset:peak_offset+1] = True
        basin_mask = np.logical_and(basin_mask, v_norm_instance < 0.8 * v_peak)

        # set pre-/post-masks by using offset arguments
        pre_length = int(duration / 3)
        post_length = int(duration / 3)
        pre_onset = max(0, onset - pre_length)
        post_offset = min(sequence_length, offset + post_length)
        pre_mask = np.zeros(sequence_length, dtype=bool)
        post_mask = np.zeros(sequence_length, dtype=bool)
        pre_mask[pre_onset:onset] = True
        post_mask[offset:post_offset] = True

        dissection[instance_id, 0, pre_mask] = True
        dissection[instance_id, 1, rise_mask] = True
        dissection[instance_id, 2, peak_mask] = True
        dissection[instance_id, 3, basin_mask] = True
        dissection[instance_id, 4, fall_mask] = True
        dissection[instance_id, 5, post_mask] = True

        # make sure pre and post doesn't overlap with any preceeding/succeeding events
        any_mask = np.any(dissection[instance_id, 1:5], axis=0)
        dissection[instance_id, 0, any_mask] = False
        dissection[instance_id, 5, any_mask] = False

        try:
            assert np.isin(np.sum(dissection[instance_id, 1:], axis=0), [0, 1]).all()
            assert np.isin(np.sum(dissection[instance_id, :-1], axis=0), [0, 1]).all()
        except:
            faulty_idx = np.where(~np.isin(np.sum(dissection[instance_id], axis=0), [0, 1]))[0]
            print(faulty_idx)
            print(dissection[instance_id, :, faulty_idx])
            print(f'{instance_id = }')
            print(f'{row_idx = }')
            breakpoint()

    return dissection


class SegmentationEvaluator(Evaluator):

    def evaluate_explainer_on_single_fold(self, A, explainer, model, fold, config, paths):
        data_key = config.dataset.key
        model_key = config.model.key
        explainer_key = config.explainer.key

        for segmentation_key in tqdm(self.keys.segmentation):
            print("Selected segmentation:", segmentation_key)

            binning_match = segmentation_binning_regex.match(segmentation_key)
            dissection_match = segmentation_dissection_regex.match(segmentation_key)
            if binning_match:
                base_segmentation_key = binning_match.group('base_segmentation_key')
                binning_property = binning_match.group('binning_property')
                n_bins = int(binning_match.group('n_bins'))
                dissection_method = None

            elif dissection_match:
                base_segmentation_key = dissection_match.group('base_segmentation_key')
                binning_property = None
                dissection_method = dissection_match.group('dissection_method')

            else:
                base_segmentation_key = segmentation_key
                binning_property = None
                dissection_method = None

            segmentation_filepath = paths.repository / 'events' / f'{base_segmentation_key}.npy'
            S = np.load(segmentation_filepath)

            if binning_property is not None:
                event_property_filepath = paths.repository / 'events' / f'{base_segmentation_key}.csv'
                df = pd.read_csv(event_property_filepath)

                if binning_property == 'v_peak':
                    df['v_peak'] = df['v_peak'].clip(upper=1000)

                if binning_property == 'duration':
                    df['duration'] = df['offset'] - df['onset']

                if binning_property == 'amplitude':
                    df['amplitude'] = np.linalg.norm(df[['amp_x', 'amp_y']].values, axis=1)

                bins, S = do_property_binning(S, df, binning_property, n_bins)

            elif dissection_method is not None:
                assert dissection_method == 'vpeak80'

                event_property_filepath = paths.repository / 'events' / f'{base_segmentation_key}.csv'
                df = pd.read_csv(event_property_filepath)
                df['v_peak'] = df['v_peak'].clip(upper=1000)

                # workaround to account for the fact that instance ids in df refer to the whole X
                X_prop = np.tile(fold.X_test[0], (S.shape[0], 1, 1))
                X_prop[fold.idx_test] = fold.X_test

                S = do_event_dissection(X_prop, S, df)

            else:
                S = np.expand_dims(S, axis=1)

            S_fold = S[fold.idx_test]

            for metric_key in tqdm(self.keys.metric):
                print("Selected metric:", metric_key)

                # load config for metric
                metric_config = Config(
                    data_key=config.dataset.key,
                    model_key=config.model.key,
                    explainer_key=config.explainer.key,
                    metric_key=metric_key,
                    segmentation_key=segmentation_key,
                    augmentation_kwargs={
                        'n_augmentations': config.augmentation.n_augmentations,
                        'noise_std': config.augmentation.noise_std,
                    },
                    gpu_id=config.gpu_id,
                    n_workers=config.n_workers,
                )
                metric_paths = ExperimentPaths(
                    basepath=paths.base, config=metric_config,
                )

                metric_scores_list = Parallel(n_jobs=config.n_workers)(
                    delayed(self.evaluate_segmentation_on_single_fold)(
                        X=fold.X_test,
                        Y=fold.Y_test,
                        A=A,
                        S=S_fold[:, [property_id], :],
                        config=metric_config,
                        paths=metric_paths,
                    )
                    for property_id in range(S_fold.shape[1])
                )

                if len(metric_scores_list) == 1:
                    metric_scores = metric_scores_list[0]

                else:
                    metric_scores = np.zeros((
                        metric_scores_list[0].shape[0],
                        len(metric_scores_list),
                    ))

                    for property_id, metric_scores_property in enumerate(metric_scores_list):
                        metric_scores[:, property_id] = metric_scores_property

                results = {
                    'idx': fold.idx_test,
                    'scores': metric_scores,
                }

                self.set_results(keys=metric_config.keys, result=results, fold=fold)

    def evaluate_segmentation_on_single_fold(self, X, Y, A, S, config, paths):
        metric_scores_property = self.evaluate_attribution_localization_with_metric(
            X=np.asarray(X),
            Y=np.asarray(Y),
            A=np.asarray(A),
            S=np.asarray(S),
            explainer=None,
            model=None,
            config=config,
            paths=paths,
        )
        return metric_scores_property

    def evaluate_attribution_localization_with_metric(
            self, X, Y, A, S, explainer, model, config, paths
    ):
        init_kwargs = config.metric['init_kwargs'].copy()
        call_kwargs = config.metric['call_kwargs'].copy()

        # This property may be needed for running eval
        sequence_length = X.shape[2]
        n_channels = X.shape[1]
        input_size = X[0].size

        for key, value in init_kwargs.items():

            # some metrics need a weighting depending on input size.
            # use monocular (dx and dy channel) 1000 as baseline length.
            if isinstance(value, str) and value == '$dataset_factor':
                baseline_size = 2 * 1000
                x_instance_size = X[0].size
                init_kwargs[key] = x_instance_size // baseline_size

            # Evaluate expression if desired and set as value.
            if isinstance(value, tuple) and value[0] == eval:
                init_kwargs[key] = eval(value[1])

        # Initialize metric.
        metric = config.metric['class'](**init_kwargs)

        # We take the maximum of all channels, as done in the original concept influence paper.
        A = np.max(A, axis=1, keepdims=True)

        # Call the metric instance to produce scores.
        scores = metric(
            model=model,
            x_batch=X,
            y_batch=np.argmax(Y, axis=1),
            a_batch=A,
            s_batch=S,
            explain_func=None,
            channel_first=True,
            device=None,
            **call_kwargs,
        )

        scores = np.array(scores)
        #print('score mean:', np.nanmean(scores))
        #print('score std:', np.nanstd(scores))
        return scores

    def finalize_experiment_results(
            self,
            results,
            experiment_keys: ExperimentKeys,
            basepath: Path,
            indices: List[int] = None,
    ) -> np.ndarray:

        if indices is not None:
            raise ValueError()

        # create numpy array for scores of all folds
        first_fold_id = list(results.keys())[0]
        n_instances = results[first_fold_id]['idx'].shape[0]
        score_shape = results[first_fold_id]['scores'].shape[1:]
        scores = np.zeros((n_instances, *score_shape)) * np.nan

        # put fold results in correct places
        for fold_id, fold_results in results.items():
            fold_idx = fold_results['idx']
            scores[fold_idx] = fold_results['scores']

        # get config for getting experiment paths
        config = Config(experiment_keys=experiment_keys)
        paths = ExperimentPaths(basepath=basepath, config=config)

        # define scores dirpath
        scores_dirpath = paths.get_eval_path('attribution_localization', use_segmentation_key=True)
        scores_filepath = scores_dirpath / 'scores.npy'

        print(f'{scores_filepath=}')
        makedirs(scores_dirpath)
        np.save(scores_filepath, scores)


def main(
    data,
    fold,
    model,
    n_augmentations,
    noise_std,
    explainer,
    metric,
    segmentation,
    load_attributions,
    save_attributions,
    batch_size,
    gpu_id,
    n_workers,
) -> None:

    experiment_key_factory = ExperimentKeyFactory(
        data_key=data,
        model_key=model,
        n_augmentations=n_augmentations,
        noise_std=noise_std,
        explainer_key=explainer,
        metric_key=metric,
        segmentation_key=segmentation,
    )
    experiment_key_factory.print()

    evaluator = SegmentationEvaluator(
        experiment_key_factory=experiment_key_factory,
        load_attributions=load_attributions,
        save_attributions=save_attributions,
    )
    evaluator.evaluate(
        folds=fold,
        batch_size=batch_size,
        gpu_id=gpu_id,
        n_workers=n_workers,
    )


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='all',
                        help='set input for evaluation')
    parser.add_argument('--fold', type=int, default=None,
                        help='set fold id for evaluation')
    parser.add_argument('--model', type=str, default='all',
                        help='set model for evaluation')
    parser.add_argument('--n_augmentations', type=int, default=0,
                        help='set number of additional noise instances for evaluation')
    parser.add_argument('--noise_std', type=float, default=0.0,
                        help='set noise level for evaluation instances')
    parser.add_argument('--explainer', type=str, default='all',
                        help='set explanation method')
    parser.add_argument('--metric', type=str, default='all',
                        help='set attribution metric')
    parser.add_argument('--segmentation', type=str, default='all',
                        help='set segmentation')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='set batch size for evaluation')
    parser.add_argument('--load_attributions', type=bool, default=False,
                        help='set to load attributions')
    parser.add_argument('--save_attributions', type=bool, default=False,
                        help='set to save attributions')
    parser.add_argument('--gpu', type=int, default=0,
                        help='set gpu id for evaluation')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='set number of workers for dataloader')

    args = parser.parse_args()
    return {
        'data': args.data,
        'fold': args.fold,
        'model': args.model,
        'explainer': args.explainer,
        'metric': args.metric,
        'segmentation': args.segmentation,
        'n_augmentations': args.n_augmentations,
        'noise_std': args.noise_std,
        'batch_size': args.batch_size,
        'load_attributions': args.load_attributions,
        'save_attributions': args.save_attributions,
        'gpu_id': args.gpu,
        'n_workers': args.n_workers,
    }


if __name__ == '__main__' :
    arguments = parse_arguments()
    main(**arguments)
