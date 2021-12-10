import random

import torch
import numpy as np


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def partial_state_dict(model: torch.nn.Module, ckpt_path: str):
    pre_ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)['state_dict']
    model_ckpt = model.state_dict()
    pre_ckpt = {k: v for k, v in pre_ckpt.items() if k in model_ckpt}
    model_ckpt.update(pre_ckpt)

    return model_ckpt

def summarize_metric(output):
    # the output is list
    average_ap = []
    average_mtta = []
    average_tta_r80 = []

    for item in range(len(output)):
        average_ap.append(output[item]['ap'])
        average_mtta.append(output[item]['mtta'])
        average_tta_r80.append(output[item]['tta_r80'])

    return np.array(average_ap).mean(), np.array(average_mtta).mean(), np.array(average_tta_r80).mean()



