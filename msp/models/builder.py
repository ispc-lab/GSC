from typing import Union, Optional, Any, Dict, List
import torch.nn as nn
from mmcv.utils import Registry, Config, build_from_cfg


SGMODEL = Registry("spatialgraphs")
TGMODEL = Registry("temporalgraphs")
ATTNMODEL = Registry("temporalattns")
NEARFUTUREMODEL = Registry("nearfuture")
ACCIDENTMODEL = Registry("accidentblocks")
GATEMODEL = Registry("gates")
LOSSES = Registry("losses")
PREDICTORS = Registry("predictors")


def build(cfg: Union[Dict, List[Dict]],
          registry: Registry,
          default_args: Optional[Dict] = None) -> Any:
    """Build a module.

    Args:
        cfg: The config of modules, is is either a dict or a list of configs.
        registry: A registry the module belongs to.
        default_args: Default arguments to build the module. Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_spatial(cfg: Union[Dict, List[Dict]]) -> Any:
    """Build spatial graph"""
    return build(cfg, SGMODEL)


def build_temporal(cfg: Union[Dict, List[Dict]]) -> Any:
    """Build temporal graph"""
    return build(cfg, TGMODEL)


def build_attn(cfg: Union[Dict, List[Dict]]) -> Any:
    """Build temporal attn"""
    return build(cfg, ATTNMODEL)

def build_nearfuture(cfg: Union[Dict, List[Dict]]) -> Any:
    """Build temporal attn"""
    return build(cfg, NEARFUTUREMODEL)


def build_accident_block(cfg: Union[Dict, List[Dict]]) -> Any:
    """Build accident block"""
    return build(cfg, ACCIDENTMODEL)


def build_gate(cfg: Union[Dict, List[Dict]]) -> Any:
    """Build gate"""
    return build(cfg, GATEMODEL)


def build_loss(cfg: Union[Dict, List[Dict]]) -> Any:
    """Build loss"""
    return build(cfg, LOSSES)


def build_predictor(cfg: Union[Dict, List[Dict]]) -> Any:
    """Build model"""
    return build(cfg, PREDICTORS)




