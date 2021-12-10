from typing import Tuple, Sequence, Dict, Union, Callable
import numpy as np
import torch
from torch import Tensor
from ..builder import PIPELINES, build_pipelines


class Compose(object):

    def __init__(self, transforms: Sequence[Union[dict, Callable]]):
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_pipelines(transform)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data: Dict[str, Dict[str, Union[Tensor, np.ndarray]]]) \
            -> Dict[str, Dict[str, Union[Tensor, np.ndarray]]]:

        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string