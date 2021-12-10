import numpy as np
import os
from tqdm import tqdm
import torch
from torch.utils.data.dataset import Dataset
from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class DashCam(Dataset):

    def __init__(self,
                 root_dir: str,
                 video_list_file: str = None,
                 pipelines: dict = None):

        self.root_dir = root_dir

        self.pipelines = Compose(pipelines) if pipelines is not None else None

        self.__sample__ = open(video_list_file, "r").read().splitlines()

    def __getitem__(self, idx: int) -> dict:
        data = np.load(os.path.join(self.root_dir, self.__sample__[idx]), allow_pickle=True).item()

        data['location'][:, :, [1, 3]] /= 1280
        data['location'][:, :, [2, 4]] /= 720

        for phase in data:
            if (phase != "accident") & (phase != "graph_index"):
                data[phase] = torch.from_numpy(data[phase])
            elif phase == "graph_index":
                data[phase] = torch.tensor(data[phase]).t().contiguous()

        data['video_id'] = self.__sample__[idx]

        return self.pipelines(data) if self.pipelines is not None else data

    def __len__(self) -> int:
        return len(self.__sample__)