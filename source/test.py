data_key = 'potec'
model_key = 'del_vmin10'
explainer_key = 'lrp_epsilon_plus'
# 'integrated_gradients_mean_smoothgrad_n10_std0.1'

gpu_id = 5
n_workers = 4

random_seed = 42
clean_plotdir = True

# input keys:
#
# judo_sr1000_sl1000_lx
# judo_sr1000_sl1000_ly
# judo_sr1000_sl1000_rx
# judo_sr1000_sl1000_ry
# judo_sr1000_sl1000_lxy
# judo_sr1000_sl1000_rxy
# judo_sr1000_sl1000_bxy
# mnist1d_1000

# attributor keys:
#
# deconvolution
# deeplift
# feature_permutation
# guided_backprop
# input_x_gradient
# integrated_gradients
# saliency

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pymovements.plot import traceplot
from pymovements.plot import tsplot

from config import basepaths
from config import Config
from evaluate.attributions import AttributionGenerator
from evaluate.model import PredictionGenerator
from experiment import ExperimentKeyFactory
from paths import ExperimentPaths


experiment_key_factory = ExperimentKeyFactory(
    data_key=data_key,
    model_key=model_key,
    explainer_key=explainer_key,
)

attribution_generator = AttributionGenerator(
    experiment_key_factory=experiment_key_factory,
)

attributions = attribution_generator.evaluate(
    indices=[87498, 67969, 860],
    gpu_id=gpu_id,
    n_workers=n_workers,
)

print(attributions)

breakpoint()

print(attributions)
