import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import shutil
import torch
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

from config import basepaths
from config import Config
from datasets.loader import DataFold
from evaluate import Evaluator
from experiment import ExperimentKeyFactory
from models.trainer import train_model, create_metric_df_from_pytorch_logger
from paths import ExperimentPaths, makedirs
from plotting.training import plot_training_history


import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class TrainingEvaluator(Evaluator):

    def __init__(self, experiment_key_factory: ExperimentKeyFactory):
        super().__init__(
            experiment_key_factory=experiment_key_factory,
        )

    def prepare(
        self,
        folds,
        config: Config,
        paths: ExperimentPaths,
    ):
        self.paths = paths

        # initialize score table
        score_columns = [
            'train_loss', 'train_accuracy',
            'val_loss', 'val_accuracy',
            'test_loss', 'test_accuracy',
        ]
        scores = pd.DataFrame(
            columns=score_columns,
            index=folds.keys(),
            dtype=float,
        )
        scores.index.name = 'fold_id'
        self.scores = scores

    def get_model(
            self,
            fold: DataFold,
            config: Config,
            filepath: Path,
            basepath: Path,
    ):
        paths = self.paths

        model_filepath_fold = filepath
        model_dirpath_fold = filepath.parent
        log_dirpath_fold = paths['logs'] / f'fold_{fold.id}'
        ckpt_filepath_fold = model_dirpath_fold / f'model_fold_{fold.id}.ckpt'
        makedirs([model_dirpath_fold, log_dirpath_fold], clean=False)
        print(f'{log_dirpath_fold = }')
        print(f'{ckpt_filepath_fold = }')

        model, trainer = train_model(
            fold=fold, config=config,
            log_dirpath=log_dirpath_fold,
            random_seed=fold.id,
            basepath=basepath,
        )
        # test using trainer
        score = trainer.test(
            model=model, dataloaders=fold.test_dl, ckpt_path='best',
        )

        # save resulting trainer checkpoint
        shutil.copy(
            src=trainer.checkpoint_callback.best_model_path,
            dst=ckpt_filepath_fold,
        )

        # save resulting model state dict
        torch.save(model.state_dict(), model_filepath_fold)

        # somehow we need to move the model to the gpu again
        model.to(f'cuda:{config.gpu_id}')

        # this comes from trainer.test..
        self.scores.loc[fold.id] = score[0]

        # compute and save predictions
        # TODO: use evaluator.generate_predictions
        with torch.no_grad():
            Y_pred_fold = np.empty(fold.Y_test.shape)
            Y_pred_fold[:, :] = np.nan

            current_start_idx = 0
            for xb, yb in tqdm(fold.test_dl, desc='Testing'):
                current_batch_size = len(yb)
                current_end_idx = current_start_idx + current_batch_size

                model.eval()
                yb_pred = model(xb.to(model.device)).detach().cpu().numpy()
                Y_pred_fold[current_start_idx:current_end_idx] = yb_pred

                current_start_idx = current_end_idx

            # evaluate performance
            test_accuracy = accuracy_score(np.argmax(fold.Y_test, axis=1),
                                           np.argmax(Y_pred_fold, axis=1))
            print(f'Test accuracy (fold {fold.id}):', test_accuracy)
            n_classes = fold.Y_test.shape[1]
            print('Chance probability:', 1 / n_classes, f'({n_classes} classes)')

            # save score table (will be rewritten in each fold)
            scores_dirpath = paths.get_eval_path('model_accuracy')
            makedirs(scores_dirpath)
            scores_filepath = scores_dirpath / 'scores.csv'
            self.scores.to_csv(scores_filepath)

        # plot training history
        df_metrics = create_metric_df_from_pytorch_logger(trainer.logger)
        filename_format = f'train_history_{{metric_name}}_{fold.id}.{{ext}}'
        plot_dirpath = paths.get_plot_path('model_train_history')
        makedirs(plot_dirpath)
        plot_training_history(df_metrics=df_metrics,
                              plot_dirpath=plot_dirpath,
                              filename_format=filename_format,
                              fileformat=['png', 'svg'])

        # additionally save training history to csv in model directory
        history_dirpath = paths.get_eval_path('model_train_history')
        history_csv_filepath = history_dirpath / f'history_fold_{fold.id}.csv'
        makedirs(history_dirpath)
        df_metrics.to_csv(history_csv_filepath)

    def evaluate_model_on_single_fold(
            self,
            model,
            fold,
            config,
            paths,
    ):
        pass


def main(
        data_key: str,
        model_key: str,
        n_augmentations: int,
        noise_std: float,
        gpu_id: int,
        n_workers: int,
        basepath: Path,
):
    experiment_key_factory = ExperimentKeyFactory(
        data_key=data_key,
        model_key=model_key,
        n_augmentations=n_augmentations,
        noise_std=noise_std,
    )
    experiment_key_factory.print()

    evaluator = TrainingEvaluator(
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
                        help='set data for training')
    parser.add_argument('--model', type=str, default='all',
                        help='set model for training')
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
        'n_augmentations': args.n_augmentations,
        'noise_std': args.noise_std,
        'gpu_id': args.gpu,
        'n_workers': args.n_workers,
    }


if __name__ == '__main__' :
    args = parse_args()
    main(basepath=basepaths.workspace, **args)
