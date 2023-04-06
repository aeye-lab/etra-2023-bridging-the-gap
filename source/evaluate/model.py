import argparse
from pathlib import Path
from typing import Dict
from typing import List

import numpy as np
import pandas as pd
import shutil
import torch
from sklearn.metrics import accuracy_score

from config import basepaths
from config import Config
from datasets.loader import DataFold
from evaluate import Evaluator
from evaluate import generate_predictions
from experiment import ExperimentKeyFactory
from experiment import ExperimentKeys
from models.trainer import train_model, create_metric_df_from_pytorch_logger
from paths import ExperimentPaths, makedirs
from plotting.training import plot_training_history


class ModelEvaluator(Evaluator):

    def __init__(self, experiment_key_factory: ExperimentKeyFactory):
        super().__init__(
            experiment_key_factory=experiment_key_factory,
        )

    def evaluate_model_on_single_fold(self, model, fold, config, paths):

        for metric_key in self.keys.metric:
            print("Selected metric:", metric_key)

            # load config for explainer only (explainer and metric configs will differ)
            metric_config = Config(
                data_key=config.dataset.key,
                model_key=config.model.key,
                explainer_key='/',
                metric_key=metric_key,
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

            if metric_key == 'accuracy':
                metric_scores = self.evaluate_model_accuracy(
                    model=model,
                    fold=fold,
                    config=metric_config,
                    paths=metric_paths,
                )
            elif metric_key.startswith('region'):
                metric_scores = self.evaluate_model_with_metric(
                    model=model,
                    fold=fold,
                    config=metric_config,
                    paths=metric_paths,
                )

            results = {
                'idx': fold.idx_test,
                'scores': metric_scores,
            }

            self.set_results(keys=metric_config.keys, result=results, fold=fold)

    def evaluate_model_with_metric(self, model, fold, config, paths):
        init_kwargs = config.metric['init_kwargs']
        call_kwargs = config.metric['call_kwargs']
        #print('metric_init_kwargs:', init_kwargs)
        #print('metric_call_kwargs:', call_kwargs)

        for key, value in init_kwargs.items():

            # some metrics need a weighting depending on input size.
            # use monocular (dx and dy channel) 1000 as baseline length.
            if value == '$dataset_factor':
                baseline_size = 2 * 1000
                x_instance_size = fold.X_test[0].size
                init_kwargs[key] = x_instance_size // baseline_size

        # Create the pixel-flipping experiment.
        metric = config.metric['class'](**init_kwargs)

        # Call the metric instance to produce scores.
        scores = metric(
            model=model,
            x_batch=fold.X_test,
            y_batch=fold.Y_test.argmax(1),
            a_batch=fold.X_test.copy(),
            channel_first=True,
            device=model.device,
            **call_kwargs,
        )

        scores = np.array(scores)
        print('mean score:', scores.mean())
        return scores

    def evaluate_model_accuracy(
            self,
            model,
            fold,
            config,
            paths,
    ):
        predictions = self.get_predictions(model, fold, apply_argmax=True)

        # evaluate performance
        test_accuracy = accuracy_score(
            np.argmax(fold.Y_test, axis=1), predictions,
        )
        return test_accuracy

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
        scores_dirpath = paths.get_eval_path('model_robustness', use_explainer_key=False)
        scores_filepath = scores_dirpath / 'scores.npy'

        print(f'{scores_filepath=}')
        makedirs(scores_dirpath)
        np.save(scores_filepath, scores)


class PredictionGenerator(Evaluator):
    """This class is handy to quickly get model predictions in a notebook."""

    def __init__(self, experiment_key_factory: ExperimentKeyFactory):
        super().__init__(
            experiment_key_factory=experiment_key_factory,
        )

    def evaluate_model_on_single_fold(
            self,
            model,
            fold,
            config,
            paths,
    ):
        preds = generate_predictions(model=model, dataloader=fold.test_dl)

        return {
            'fold': fold,
            'predictions': preds,
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

        X = np.zeros((n_instances, *x_shape)) * np.nan
        Y = np.zeros((n_instances, *y_shape)) * np.nan
        Y_pred = np.zeros((n_instances, *y_shape)) * np.nan
        fold_ids = np.zeros((n_instances, ), dtype=int) * np.nan

        # put fold results in correct places
        for fold_id, fold_results in results.items():
            fold_idx = fold_results['fold'].idx_test

            X[fold_idx] = fold_results['fold'].X_test
            Y[fold_idx] = fold_results['fold'].Y_test
            Y_pred[fold_idx] = fold_results['predictions']
            fold_ids[fold_idx] = fold_results['fold'].id

        if indices is None:
            indices = slice(None)

        return {
            #'X': X[indices],
            #'Y': Y[indices],
            'Y_pred': Y_pred[indices],
            'fold_ids': fold_ids,
        }


def main(
        data_key: str,
        model_key: str,
        metric_key: str,
        n_augmentations: int,
        noise_std: float,
        gpu_id: int,
        n_workers: int,
        basepath: Path,
):
    experiment_key_factory = ExperimentKeyFactory(
        data_key=data_key,
        model_key=model_key,
        explainer_key='/',
        metric_key=metric_key,
        n_augmentations=n_augmentations,
        noise_std=noise_std,
    )
    experiment_key_factory.print()

    evaluator = ModelEvaluator(
        experiment_key_factory=experiment_key_factory,
    )
    evaluator.evaluate(
        gpu_id=gpu_id,
        n_workers=n_workers,
        basepath=basepath,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='all',
                        help='set data for evaluation')
    parser.add_argument('--model', type=str, default='all',
                        help='set model for evaluation')
    parser.add_argument('--metric', type=str, required=True,
                        help='set metric')
    parser.add_argument('--n_augmentations', type=int, default=0,
                        help='set number of additional noise instances for training')
    parser.add_argument('--noise_std', type=float, default=0.0,
                        help='set noise level for training instances')
    parser.add_argument('--gpu', type=int, default=0,
                        help='set gpu id for training')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='set number of workers for dataloader')

    args = parser.parse_args()
    return {
        'data_key': args.data,
        'model_key': args.model,
        'metric_key': args.metric,
        'n_augmentations': args.n_augmentations,
        'noise_std': args.noise_std,
        'gpu_id': args.gpu,
        'n_workers': args.n_workers,
    }


if __name__ == '__main__' :
    args = parse_args()
    main(basepath=basepaths.workspace, **args)
