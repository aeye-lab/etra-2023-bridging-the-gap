from __future__ import annotations
from datetime import timedelta
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from attribute import compute_attributions
from attribute import Explainer
from config import basepaths
from config import Config
from datasets.loader import load_data
from datasets.loader import DataFold
from datasets.loader import DataFoldFactory
from experiment import ExperimentKeyFactory
from experiment import ExperimentKeys
from models.loader import load_model
from paths import ExperimentPaths, makedirs
from preprocessing import preprocess_datafold


def generate_predictions(model, dataloader: DataLoader):
    # First we create a list, then we concatenate it with numpy.
    preds = []

    # Compute and save predictions.
    with torch.no_grad():
        for xb, yb in tqdm(dataloader, desc='Testing'):
            model.eval()
            yb_pred = model(xb.to(model.device)).detach().cpu().numpy()
            preds.append(yb_pred)

    return np.concatenate(preds)


def reduce_data_to_indices(data: Dict[str, Any], indices: List[int],
) -> Dict[str, Any]:
    # Setup mask for all passed indices.
    n_instances = data['X'].shape[0]
    indices_mask = np.zeros(n_instances, dtype=bool)
    indices_mask[indices] = True

    # We need to create list of fold_ids because keys() will change during loop.
    fold_ids = list(data['folds'].keys())
    for fold_id in fold_ids:
        # We only need to take the intersections of indices and test masks.
        data['folds'][fold_id]['test'] = np.logical_and(
            data['folds'][fold_id]['test'], indices_mask,
        )

        if data['folds'][fold_id]['test'].sum() == 0:
            del data['folds'][fold_id]
    return data


def reduce_data_to_folds(data: Dict[str, Any], folds: Union[List[int], int],
) -> Dict[str, Any]:
    if isinstance(folds, int):
        folds = [folds]

    # Remove folds which are not desired.
    # We need to create list of fold_ids because keys() will change during loop.
    fold_ids = list(data['folds'].keys())
    for fold_id in fold_ids:
        if fold_id not in folds:
            del data['folds'][fold_id]
    return data


class Evaluator:

    def __init__(
        self,
        experiment_key_factory: ExperimentKeyFactory,
        load_attributions: bool = False,
        save_attributions: bool = False,
    ):
        self.keys = experiment_key_factory
        self.data = {}
        self.results = {}

        self.load_attributions = load_attributions
        self.save_attributions = save_attributions

    def prepare(
        self,
        folds,
        config: Config,
        paths: ExperimentPaths,
    ):
        pass

    def prepare_results(self, data_key: str, data: dict):
        self.results[data_key] = {
            model_key: {
                explainer_key: {
                    metric_key: {
                        segmentation_key: {
                            fold_id: None for fold_id in data['folds'].keys()
                        }
                        for segmentation_key in self.keys.segmentation
                    }
                    for metric_key in self.keys.metric
                }
                for explainer_key in self.keys.explainer
            }
            for model_key in self.keys.model
        }

    def finalize_all_results(self, basepath: Path, indices: List[int]):
        for experiment_keys in self.keys.iterate():
            experiment_key_results = self.get_results(experiment_keys)
            finalized_results = self.finalize_experiment_results(
                results=experiment_key_results,
                experiment_keys=experiment_keys,
                basepath=basepath,
                indices=indices,
            )
            self.set_results(keys=experiment_keys, result=finalized_results)

    def finalize_experiment_results(
            self,
            results,
            experiment_keys: ExperimentKeys,
            basepath: Path,
            indices: List[int] = None,
    ):
        return results

    def evaluate(
            self,
            gpu_id: int,
            n_workers: int,
            batch_size: Optional[int] = None,
            basepath: Optional[Path] = None,
            indices: Optional[List[int]] = None,
            use_cached_data: bool = True,
            folds: Optional[Union[List[int], int]] = None,
    ) -> None:
        if basepath is None:
            basepath = basepaths.workspace

        start_time = timer()

        for data_key in self.keys.data:
            print("Selected dataset:", data_key)

            # load config for data only (model, explainer and metric configs will differ)
            data_config = Config(
                data_key=data_key,
                augmentation_kwargs={
                    'n_augmentations': self.keys.n_augmentations,
                    'noise_std': self.keys.noise_std,
                },
                gpu_id=gpu_id,
                n_workers=n_workers,
            )
            data_paths = ExperimentPaths(basepath=basepath, config=data_config)

            if use_cached_data and data_key in self.data:
                print("Use cached data")
            else:
                print("Loading data...")
                data = load_data(config=data_config, paths=data_paths)
                self.data[data_key] = data

            if indices is not None:
                data = reduce_data_to_indices(data=data, indices=indices)

            if folds is not None:
                data = reduce_data_to_folds(data=data, folds=folds)

            self.prepare_results(data_key=data_key, data=data)

            for model_key in self.keys.model:
                print("Selected model:", model_key)

                # load config for model only (explainer and metric configs will differ)
                model_config = Config(
                    data_key=data_key,
                    model_key=model_key,
                    augmentation_kwargs={
                        'n_augmentations': self.keys.n_augmentations,
                        'noise_std': self.keys.noise_std,
                    },
                    gpu_id=gpu_id,
                    n_workers=n_workers,
                )
                model_paths = ExperimentPaths(
                    basepath=basepath, config=model_config,
                )

                self.prepare(
                    folds=data['folds'],
                    config=model_config,
                    paths=model_paths,
                )

                if batch_size is None:
                    batch_size = model_config.model['batch_size']

                datafold_factory = DataFoldFactory(
                    **data,
                    batch_size=batch_size,
                    augmentation_config=model_config.augmentation,
                    n_workers=model_config.n_workers,
                )
                self.evaluate_model_on_folds(
                    datafold_factory=datafold_factory,
                    config=model_config,
                    paths=model_paths,
                )

        self.finalize_all_results(basepath=basepath, indices=indices)

        end_time = timer()
        computation_time = end_time - start_time
        print('Total computation time:', timedelta(seconds=computation_time))

        return self.results

    def evaluate_model_on_folds(
            self,
            datafold_factory: DataFoldFactory,
            config: Config,
            paths: ExperimentPaths,
    ) -> None:
        if config.model['framework'] != 'pytorch':
            raise TypeError('model framework needs to be pytorch but is: {model_config["framework"]}')

        for fold in tqdm(datafold_factory):
            # preprocess datafold
            fold = preprocess_datafold(
                fold=fold,
                method=config.model['preprocessing'],
                **config.model.get('preprocessing_kwargs', {}),
            )

            # construct model filepath to get model
            model_dirpath_fold = paths['model']
            model_filepath_fold = model_dirpath_fold / f'model_fold_{fold.id}.pth'
            print(f'{model_filepath_fold = }')

            model = self.get_model(
                fold=fold, config=config,
                filepath=model_filepath_fold,
                basepath=paths.base,
            )

            result = self.evaluate_model_on_single_fold(
                model=model, fold=fold,
                config=config, paths=paths,
            )
            self.set_results(keys=config.keys, result=result, fold=fold)

    def evaluate_model_on_single_fold(self, model, fold, config, paths):
        predictions = self.get_predictions(model, fold)

        for explainer_key in self.keys.explainer:
            print("Selected explainer:", explainer_key)

            # load config for explainer only (explainer and metric configs will differ)
            explainer_config = Config(
                data_key=config.dataset.key,
                model_key=config.model.key,
                explainer_key=explainer_key,
                augmentation_kwargs={
                    'n_augmentations': config.augmentation.n_augmentations,
                    'noise_std': config.augmentation.noise_std,
                },
                gpu_id=config.gpu_id,
                n_workers=config.n_workers,
            )
            explainer_paths = ExperimentPaths(
                basepath=paths.base, config=explainer_config,
            )

            explainer = Explainer(
                model=model,
                config=explainer_config,
                validation_data=fold.X_val,
            )
            A_test = self.get_explanations(
                model=model, fold=fold,
                predictions=predictions,
                config=explainer_config,
                paths=explainer_paths,
            )
            result = self.evaluate_explainer_on_single_fold(
                A=A_test,
                explainer=explainer,
                model=model,
                fold=fold,
                config=explainer_config,
                paths=explainer_paths,
            )

            if result is not None:
                self.set_results(keys=explainer_config.keys, result=result, fold=fold)

    def evaluate_explainer_on_single_fold(self, A, explainer, model, fold, config, paths) -> object:
        data_key = config.dataset.key
        model_key = config.model.key
        explainer_key = config.explainer.key

        for metric_key in self.keys.metric:
            print("Selected metric:", metric_key)

            # load config for explainer only (explainer and metric configs will differ)
            metric_config = Config(
                data_key=config.dataset.key,
                model_key=config.model.key,
                explainer_key=config.explainer.key,
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

            metric_scores = self.evaluate_attributions_with_metric(
                X=fold.X_test, Y=fold.Y_test, A=A,
                explainer=explainer,
                model=model,
                config=metric_config,
                paths=metric_paths,
            )
            results = {
                'idx': fold.idx_test,
                'scores': metric_scores,
            }
            self.set_results(keys=metric_config.keys, result=results, fold=fold)

    def get_model(self, fold, config, filepath, basepath,
    ) -> torch.nn.Module:
        return load_model(
            fold=fold, config=config, filepath=filepath, basepath=basepath,
        )

    def get_explanations(self, model, fold, predictions, config, paths) -> np.ndarray:
        attributions_dirpath = paths.get_eval_path('attributions')
        attributions_filepath = attributions_dirpath / f'fold_{fold.id}.npy'

        if self.load_attributions:
            print('load attributions')
            print(f'{attributions_filepath = }')
            A = np.load(attributions_filepath)
        else:
            A = compute_attributions(
                explainer_config=config.explainer,
                model_config=config.model,
                model=model,
                dataloader=fold.test_dl,
                val_data=fold.X_val,
                predictions=predictions,
            )

        if self.save_attributions:
            print('save attributions')
            print(f'{attributions_filepath = }')
            makedirs(attributions_dirpath)
            np.save(attributions_filepath, A)

        return A

    def get_predictions(self, model, fold, apply_argmax: bool = True):
        predictions = np.zeros(fold.Y_test.shape, dtype=fold.Y_test.dtype) * np.nan
        current_start_idx = 0
        for xb, yb in fold.test_dl:
            current_batch_size = len(yb)
            current_end_idx = current_start_idx + current_batch_size

            model.eval()
            yb_pred = model(xb.to(model.device)).detach().cpu().numpy()
            predictions[current_start_idx:current_end_idx] = yb_pred
            current_start_idx = current_end_idx

        if apply_argmax:
            predictions = np.argmax(predictions, axis=1)
        return predictions

    def get_results(self, keys: ExperimentKeys):
        results_dict = self.results[keys.data]

        if keys.model is None:
            return results_dict
        results_dict = results_dict[keys.model]

        if keys.explainer is None:
            return results_dict
        results_dict = results_dict[keys.explainer]

        if keys.metric is None:
            return results_dict
        results_dict = results_dict[keys.metric]

        if keys.segmentation is None:
            return results_dict
        results_dict = results_dict[keys.segmentation]

        return results_dict

    def set_results(self, keys: ExperimentKeys, result: Any,
                    fold: Optional[DataFold] = None):
        if result is None:
            return

        results_dict = self.results

        if keys.model is None:
            if fold:
                results_dict[keys.data][fold.id] = result
                return
            else:
                results_dict[keys.data] = result
                return
        else:
            results_dict = results_dict[keys.data]

        if keys.explainer is None:
            if fold:
                results_dict[keys.model][fold.id] = result
                return
            else:
                results_dict[keys.model] = result
                return
        else:
            results_dict = results_dict[keys.model]

        if keys.metric is None:
            if fold:
                results_dict[keys.explainer][fold.id] = result
                return
            else:
                results_dict[keys.explainer] = result
                return
        else:
            results_dict = results_dict[keys.explainer]

        if keys.segmentation is None:
            if fold:
                results_dict[keys.metric][fold.id] = result
                return
            else:
                results_dict[keys.metric] = result
                return

        else:
            results_dict = results_dict[keys.metric]
            if fold:
                results_dict[keys.segmentation][fold.id] = result
                return
            else:
                results_dict[keys.segmentation] = result
                return
