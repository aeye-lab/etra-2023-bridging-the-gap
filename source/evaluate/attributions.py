import os
n_threads = "4"

#os.environ["NVIDIA_VISIBLE_DEVICES"] = "3"
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["OMP_NUM_THREADS"] = n_threads
os.environ["OPENBLAS_NUM_THREADS"] = n_threads
os.environ["MKL_NUM_THREADS"] = n_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = n_threads
os.environ["NUMEXPR_NUM_THREADS"] = n_threads

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from config import Config
from evaluate import Evaluator
from experiment import ExperimentKeyFactory, ExperimentKeys
from paths import ExperimentPaths, makedirs


class AttributionEvaluator(Evaluator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate_attributions_with_metric(self, X, Y, A, explainer, model, config, paths):
        init_kwargs = config.metric['init_kwargs'].copy()
        call_kwargs = config.metric['call_kwargs'].copy()
        #print('metric_init_kwargs:', init_kwargs)
        #print('metric_call_kwargs:', call_kwargs)

        for key, value in init_kwargs.items():

            # some metrics need a weighting depending on input size.
            # use monocular (dx and dy channel) 1000 as baseline length.
            if isinstance(value, str) and value == '$dataset_factor':
                baseline_size = 2 * 1000
                x_instance_size = X[0].size
                init_kwargs[key] = x_instance_size // baseline_size

        # Create the pixel-flipping experiment.
        metric = config.metric['class'](**init_kwargs)

        # We take the mean of all channels. A lot of Quantus metric don't
        # yet support multi-channel time series attributions.
        A = np.mean(A, axis=1, keepdims=True)

        # Call the metric instance to produce scores.
        scores = metric(
            model=model,
            x_batch=X,
            y_batch=np.argmax(Y, axis=1),
            a_batch=A,
            explain_func=explainer.explain_batch,
            channel_first=True,
            device=model.device,
            **call_kwargs,
        )

        scores = np.array(scores)
        print('mean score:', scores.mean())
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
        scores_dirpath = paths.get_eval_path('attribution_metrics')
        scores_filepath = scores_dirpath / 'scores.npy'

        print(f'{scores_filepath=}')
        makedirs(scores_dirpath)
        np.save(scores_filepath, scores)


class AttributionGenerator(Evaluator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate_explainer_on_single_fold(self, A, explainer, model, fold, config, paths):
        return {
            'fold': fold,
            'attributions': A,
        }

    def finalize_experiment_results(
            self,
            results,
            experiment_keys: ExperimentKeys,
            basepath: Path,
            indices: List[int] = None,
    ) -> np.ndarray:
        # create numpy array for scores of all folds
        first_fold_id = list(results.keys())[0]
        n_instances = results[first_fold_id]['fold'].idx_test.shape[0]

        x_shape = results[first_fold_id]['fold'].X_test.shape[1:]
        y_shape = results[first_fold_id]['fold'].Y_test.shape[1:]
        a_shape = results[first_fold_id]['attributions'].shape[1:]

        X = np.zeros((n_instances, *x_shape)) * np.nan
        Y = np.zeros((n_instances, *y_shape)) * np.nan
        A = np.zeros((n_instances, *a_shape)) * np.nan
        fold_ids = np.zeros((n_instances, ), dtype=int) * np.nan

        # put fold results in correct places
        for fold_id, fold_results in results.items():
            fold_idx = fold_results['fold'].idx_test

            X[fold_idx] = fold_results['fold'].X_test
            Y[fold_idx] = fold_results['fold'].Y_test
            A[fold_idx] = fold_results['attributions']
            fold_ids[fold_idx] = fold_results['fold'].id

        if indices is None:
            indices = slice(None)

        return {
            #'X': X[indices],
            #'Y': Y[indices],
            'A': A[indices],
            'fold_ids': fold_ids[indices],
        }


def main(
    data,
    fold,
    model,
    n_augmentations,
    noise_std,
    explainer,
    metric,
    load_attributions,
    save_attributions,
    save_only,
    batch_size,
    gpu_id,
    n_workers,
) -> None:

    if save_only:
        load_attributions = False
        save_attributions = True

        experiment_key_factory = ExperimentKeyFactory(
            data_key=data,
            model_key=model,
            n_augmentations=n_augmentations,
            noise_std=noise_std,
            explainer_key=explainer,
        )
        experiment_key_factory.print()

        evaluator = AttributionGenerator(
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
    else:
        experiment_key_factory = ExperimentKeyFactory(
            data_key=data,
            model_key=model,
            n_augmentations=n_augmentations,
            noise_std=noise_std,
            explainer_key=explainer,
            metric_key=metric,
        )
        experiment_key_factory.print()

        evaluator = AttributionEvaluator(
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
    parser.add_argument('--batch_size', type=int, default=64,
                        help='set batch size for evaluation')
    parser.add_argument('--load_attributions', type=bool, default=False,
                        help='set to load attributions')
    parser.add_argument('--save_attributions', type=bool, default=False,
                        help='set to save attributions')
    parser.add_argument('--save_only', type=bool, default=False,
                        help='set to only save attributions without metric')
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
        'n_augmentations': args.n_augmentations,
        'noise_std': args.noise_std,
        'batch_size': args.batch_size,
        'load_attributions': args.load_attributions,
        'save_attributions': args.save_attributions,
        'save_only': args.save_only,
        'gpu_id': args.gpu,
        'n_workers': args.n_workers,
    }


if __name__ == '__main__' :
    arguments = parse_arguments()
    main(**arguments)
