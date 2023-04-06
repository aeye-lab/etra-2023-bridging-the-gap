import torch

import zennit.composites

from zennit import composites as zennit_composites
from zennit.composites import register_composite, LayerMapComposite, SpecialFirstLayerMapComposite
from zennit.composites import layer_map_base
from zennit.core import Composite
from zennit.rules import AlphaBeta, Gamma, Epsilon, ZPlus, Flat
from zennit.types import Convolution, Linear, AvgPool, Activation
import torch


class NameMapComposite(Composite):
    '''A Composite for which hooks are specified by a mapping from module types to hooks.

    Parameters
    ----------
    name_map: `list[tuple[tuple[str, ...], Hook]]`
        A mapping as a list of tuples, with a tuple of applicable module names and a Hook.
    layer_map: list[tuple[tuple[torch.nn.Module, ...], Hook]], optional
        A mapping as a list of tuples, with a tuple of applicable module types and a Hook.
    canonizers: list[:py:class:`zennit.canonizers.Canonizer`], optional
        List of canonizer instances to be applied before applying hooks.
    '''
    def __init__(self, name_map, layer_map=None, canonizers=None):
        self.name_map = name_map
        self.layer_map = layer_map if layer_map is not None else ()
        super().__init__(module_map=self.mapping, canonizers=canonizers)

    # pylint: disable=unused-argument
    def mapping(self, ctx, name, module):
        '''Get the appropriate hook given a mapping from module names to hooks.

        Parameters
        ----------
        ctx: dict
            A context dictionary to keep track of previously registered hooks.
        name: str
            Name of the module.
        module: obj:`torch.nn.Module`
            Instance of the module to find a hook for.

        Returns
        -------
        obj:`Hook` or None
            The hook found with the module name in the given name map, or with
            the module type in the given layer map, or None if no applicable
            hook was found.
        '''
        return next(
            (hook for names, hook in self.name_map if name in names),
            next(
                (hook for types, hook in self.layer_map
                 if isinstance(module, types)),
                None,
            ),
        )


@register_composite('epsilon_alpha_beta')
class EpsilonAlphaBeta(LayerMapComposite):
    '''An explicit composite using the alpha2-beta1 rule for all convolutional layers and the epsilon rule for all
    fully connected layers.
    Parameters
    ----------
    epsilon: callable or float, optional
        Stabilization parameter for the ``Epsilon`` rule. If ``epsilon`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator. Note that this is
        called ``stabilizer`` for all other rules.
    stabilizer: callable or float, optional
        Stabilization parameter for rules other than ``Epsilon``. If ``stabilizer`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator.
    layer_map: list[tuple[tuple[torch.nn.Module, ...], Hook]]
        A mapping as a list of tuples, with a tuple of applicable module types and a Hook. This will be prepended to
        the ``layer_map`` defined by the composite.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.
    canonizers: list[:py:class:`zennit.canonizers.Canonizer`], optional
        List of canonizer instances to be applied before applying hooks.
    '''
    def __init__(self, epsilon=1e-6, alpha=2, beta=1, stabilizer=1e-6, layer_map=None, zero_params=None, canonizers=None):
        if layer_map is None:
            layer_map = []

        rule_kwargs = {'zero_params': zero_params}
        layer_map = layer_map + layer_map_base(stabilizer) + [
            (Convolution, AlphaBeta(alpha=alpha, beta=beta, stabilizer=stabilizer, **rule_kwargs)),
            (torch.nn.Linear, Epsilon(epsilon=epsilon, **rule_kwargs)),
        ]
        super().__init__(layer_map=layer_map, canonizers=canonizers)


@register_composite('epsilon_alpha2_beta1_flat')
class EpsilonAlphaBetaFlat(SpecialFirstLayerMapComposite):
    '''An explicit composite using the flat rule for any linear first layer, the alpha2-beta1 rule for all other
    convolutional layers and the epsilon rule for all other fully connected layers.
    Parameters
    ----------
    epsilon: callable or float, optional
        Stabilization parameter for the ``Epsilon`` rule. If ``epsilon`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator. Note that this is
        called ``stabilizer`` for all other rules.
    stabilizer: callable or float, optional
        Stabilization parameter for rules other than ``Epsilon``. If ``stabilizer`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator.
    layer_map: list[tuple[tuple[torch.nn.Module, ...], Hook]]
        A mapping as a list of tuples, with a tuple of applicable module types and a Hook. This will be prepended to
        the ``layer_map`` defined by the composite.
    first_map: `list[tuple[tuple[torch.nn.Module, ...], Hook]]`
        Applicable mapping for the first layer, same format as `layer_map`. This will be prepended to the ``first_map``
        defined by the composite.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.
    canonizers: list[:py:class:`zennit.canonizers.Canonizer`], optional
        List of canonizer instances to be applied before applying hooks.
    '''
    def __init__(self, epsilon=1e-6, alpha=2, beta=1, stabilizer=1e-6, layer_map=None, first_map=None, zero_params=None, canonizers=None):
        if layer_map is None:
            layer_map = []
        if first_map is None:
            first_map = []

        rule_kwargs = {'zero_params': zero_params}
        layer_map = layer_map + layer_map_base(stabilizer) + [
            (Convolution, AlphaBeta(alpha=alpha, beta=beta, stabilizer=stabilizer, **rule_kwargs)),
            (torch.nn.Linear, Epsilon(epsilon=epsilon, **rule_kwargs)),
        ]
        first_map = first_map + [
            (Linear, Flat(stabilizer=stabilizer, **rule_kwargs))
        ]
        super().__init__(layer_map=layer_map, first_map=first_map, canonizers=canonizers)


@register_composite('epsilon_gamma')
class EpsilonGamma(LayerMapComposite):
    '''An explicit composite using the gamme rule for all convolutional layers and the epsilon rule for all fully
    connected layers.
    '''
    def __init__(self, gamma=0.1, epsilon=1e-6, stabilizer=1e-6, layer_map=None, canonizers=None):
        if layer_map is None:
            layer_map = []

        layer_map = layer_map + layer_map_base(stabilizer) + [
            (Convolution, Gamma(gamma=gamma)),
            (torch.nn.Linear, Epsilon(epsilon=epsilon)),
        ]
        super().__init__(layer_map, canonizers=canonizers)


@register_composite('epsilon_gamma_flat')
class EpsilonGammaFlat(SpecialFirstLayerMapComposite):
    '''An explicit composite using the flat rule for the first layer, the gamma rule for all convolutional layers and
    the epsilon rule for all fully connected layers.
    '''
    def __init__(self, gamma=0.1, epsilon=1e-6, stabilizer=1e-6, layer_map=None, first_map=None, canonizers=None):
        if layer_map is None:
            layer_map = []
        if first_map is None:
            first_map = []

        layer_map = layer_map + layer_map_base(stabilizer) + [
            (Convolution, Gamma(gamma=gamma)),
            (torch.nn.Linear, Epsilon(epsilon=epsilon)),
        ]
        first_map = first_map + [
            (Linear, Flat(stabilizer=stabilizer))
        ]
        super().__init__(layer_map=layer_map, first_map=first_map, canonizers=canonizers)


@register_composite('epsilon_only')
class EpsilonOnly(LayerMapComposite):
    '''An explicit composite using the epsilon rule for all layers.
    '''
    def __init__(self, epsilon=1e-6, stabilizer=1e-6, canonizers=None):
        layer_map = zennit.composites.layer_map_base(stabilizer) + [
            (Convolution, Epsilon(epsilon=epsilon)),
            (Linear, Epsilon(epsilon=epsilon)),
            (torch.nn.Linear, Epsilon(epsilon=epsilon)),
        ]
        super().__init__(layer_map, canonizers=canonizers)


@register_composite('epsilon_only_dense')
class EpsilonOnlyDense(LayerMapComposite):
    '''An explicit composite using the epsilon rule for all layers.
    '''
    def __init__(self, canonizers=None):
        layer_map = zennit_composites.LAYER_MAP_BASE + [
            #(Convolution, Epsilon()),
            (torch.nn.Linear, Epsilon()),
        ]
        super().__init__(layer_map, canonizers=canonizers)


@register_composite('epsilon_only_conv')
class EpsilonOnlyConv(LayerMapComposite):
    '''An explicit composite using the epsilon rule for all layers.
    '''
    def __init__(self, canonizers=None):
        layer_map = zennit_composites.LAYER_MAP_BASE + [
            (Convolution, Epsilon()),
            #(torch.nn.Linear, Epsilon()),
        ]
        super().__init__(layer_map, canonizers=canonizers)


@register_composite('epsilon_flat')
class EpsilonFlat(SpecialFirstLayerMapComposite):
    '''An explicit composite using the flat rule for the first layer and the epsilon rule for all subsequent layers.
    '''
    def __init__(self, canonizers=None):
        layer_map = zennit_composites.LAYER_MAP_BASE + [
            (Convolution, Epsilon()),
            (torch.nn.Linear, Epsilon()),
        ]
        first_map = [
            (Linear, Flat())
        ]
        super().__init__(layer_map, first_map, canonizers=canonizers)


@register_composite('epsilon_zplus_flat')
class EpsilonZPlusFlat(SpecialFirstLayerMapComposite):
    '''An explicit composite using the flat rule for the first layer, the zplus rule for all convolutional layers and
    the epsilon rule for all fully connected layers.
    '''
    def __init__(self, canonizers=None):
        layer_map = zennit_composites.LAYER_MAP_BASE + [
            (Convolution, ZPlus()),
            (torch.nn.Linear, Epsilon()),
        ]
        first_map = [
            (Linear, Flat())
        ]
        super().__init__(layer_map, first_map, canonizers=canonizers)
