import os
import sys
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)

from typing import List, Dict, Tuple
import torch
from src.models.task_vectors import TaskVector
from src.models.modeling import ImageEncoder

def emr_apply_vector(vector, pretrained_encoder: ImageEncoder):#, scaling_coef=1.0):
    """Apply a task vector to a pretrained model."""
    with torch.no_grad():
        pretrained_state_dict = pretrained_encoder.state_dict()
        new_state_dict = {}
        for key in pretrained_state_dict:
            if key not in vector:
                # import ipdb; ipdb.set_trace()
                print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                continue
            new_state_dict[key] = pretrained_state_dict[key] + vector[key]
    pretrained_encoder.load_state_dict(new_state_dict, strict=False)
    return pretrained_encoder


def emr_merge(task_vectors: List[TaskVector])->Tuple[Dict[str, torch.Tensor],
                                                      Dict[str, torch.Tensor],
                                                      torch.Tensor]:
    sum_param = {}
    n2p = []
    for m in range(len(task_vectors)):
        n2p_temp = task_vectors[m].vector
        n2p.append(n2p_temp)
        for n in n2p_temp:
            if n not in sum_param:
                sum_param[n] = []
            sum_param[n].append(n2p_temp[n])
    sum_param = {k: torch.stack(v, 0).mean(0) for k, v in sum_param.items()}
    vector_unified = {}
    scales = torch.zeros(len(task_vectors))
    masks = {}
    for n in sum_param:
        masks[n] = []
        flag = (sum_param[n]>0) * 2 - 1
        param_max = torch.zeros_like(n2p[0][n])
        for m in range(len(task_vectors)):
            param = task_vectors[m].vector[n]
            mask = (param * flag) > 0
            masks[n].append(mask)
            param_abs = torch.abs(mask*param)
            param_max = torch.where(param_abs>param_max, param_abs, param_max)
            # import ipdb; ipdb.set_trace()
            scales[m] += torch.mean(torch.abs(param))
        vector_unified[n] =  param_max * flag
    new_scales = torch.zeros(len(task_vectors))
    for m in range(len(task_vectors)):
        for n in vector_unified:
            p = vector_unified[n] * masks[n][m]
            new_scales[m] += torch.mean(torch.abs(p))
    rescalers = scales / new_scales

    return vector_unified, masks, rescalers