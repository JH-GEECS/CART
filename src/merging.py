import sys
import os

repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)

from src.dataset.common import (
    get_dataloader,
    maybe_dictionarize,
    get_dataloader_shuffle,
)
from src.dataset.registry import get_dataset
from src.eval import eval_single_dataset
from src.models.heads import get_classification_head
from src.models.modeling import ImageEncoder
from src.models.task_vectors import TaskVector
from src.utils import is_TA_mode, get_dir_dict, garbage_collect
from src.logger import ExperimentResult

from typing import List, Optional, Union, Dict

import ray
import glob

import argparse
import csv
import wandb
from copy import deepcopy
from itertools import cycle
import functools

import torch
import numpy as np
from torch.func import functional_call
from torch import nn, optim
from torch.nn.utils import stateless
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from omegaconf import OmegaConf, DictConfig

import pandas as pd

CPU_DEVICE = "cpu"

def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def straight_through_mask(mat_svd, mask):
    U, s, V_T = mat_svd
    s_masked = mask * s + (((mask > 0.5).float() - mask) * s).detach()
    return U @ torch.diag(s_masked) @ V_T

def load_config(config_path: str) -> DictConfig:
    config = OmegaConf.load(config_path)
    if "device" in config and config.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
    return config

class StaticMergeModule(nn.Module):
    def __init__(
        self, config, zero_shot_encoder: ImageEncoder, task_vectors: List[TaskVector]
    ):
        super(StaticMergeModule, self).__init__()
        self.config = config
        self.pretrained_model = zero_shot_encoder

        self.task_vectors = task_vectors

        self.exam_datasets = config.tasks
        self.device = config.device


    def _get_truncated_task_vectors(self, rank_ratio: Union[List[int], float] = None,
                                    prev_origin: Optional[Dict[str, torch.Tensor]] = None,
                                    new_origin: Optional[Dict[str, torch.Tensor]] = None,
                                    get_decomposed_tv: bool = False,
                                    scalar_expansion_list: Optional[List[float]] = None,
                                    energy_threshold: Optional[bool] = None,
                                    specific_singular_val: Optional[float] = None,
                                    discrete_rank: Optional[bool] = False,
                                    ) -> Union[List[TaskVector], List[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]]:
        if energy_threshold and discrete_rank:
            raise ValueError(
                f"only one of energy_threshold and discrete_rank should be True, currently energy_threshold:{energy_threshold}, discrete_rank:{discrete_rank}"
            )
        
        if discrete_rank:
            rank_ratio = OmegaConf.to_container(rank_ratio, resolve=True)
            assert isinstance(rank_ratio, list) and all(isinstance(x, int) for x in rank_ratio), \
                f"rank_ratio should be List[int], currently {type(rank_ratio)}, {rank_ratio}"
            print(f"discrete rank selection mode enabled with {rank_ratio}")
        else:
            assert type(rank_ratio) == float, "rank_ratio should be float"

        if specific_singular_val:
            assert discrete_rank, "max singular value option is only thoroughly considered in the case of discrete rank selection"

        move_origin = False
        if prev_origin and new_origin:
            move_origin = True

        print(f"Truncating task vectors with rank ratio: {rank_ratio}")

        if scalar_expansion_list:
            assert move_origin, "scalar expansion should be used with origin move"
            assert len(scalar_expansion_list) == len(self.task_vectors), "scalar expansion list should have same length with task vectors"

            scalar_expansion_coeff_list = len(self.task_vectors) * np.array(scalar_expansion_list)
            print(f"expanding theta_i by {scalar_expansion_coeff_list}")
        decomposed_task_vectors = []
        for t_idx, task_vector in tqdm(enumerate(self.task_vectors), total=len(self.task_vectors), desc="Truncating task vectors"):
            svd_vector = {}
            for key, value in task_vector.vector.items():
                if ("attn" in key or "mlp" in key) and not ("ln" in key or "bias" in key):
                    if move_origin:
                        if scalar_expansion_list:
                            scalar_expansion_coeff = scalar_expansion_coeff_list[t_idx]
                            _val = scalar_expansion_coeff * (value + prev_origin[key]) - new_origin[key]
                        else:
                            _val = (value + prev_origin[key] - new_origin[key])
                    else:
                        _val = value

                    U, s, V_T = torch.linalg.svd(_val.to(self.device), full_matrices=False)
                    dim = s.shape[0]

                    if discrete_rank:
                        rank_list: List[int] = rank_ratio
                        if get_decomposed_tv:
                            if specific_singular_val is not None:
                                _s = torch.ones_like(s[rank_list]) * specific_singular_val
                            else:
                                _s = s[rank_list].to(CPU_DEVICE)

                            svd_vector[key] = {
                                "U": U[:, rank_list].to(CPU_DEVICE),
                                "s": _s,
                                "V_T": V_T[rank_list, :].to(CPU_DEVICE),
                            }
                        else:
                            if specific_singular_val is not None:
                                _s = torch.ones_like(s[rank_list]) * specific_singular_val
                            else:
                                _s = s[rank_list]

                            recon = (
                                (U[:, rank_list] * _s) @ V_T[rank_list, :]
                            )
                            svd_vector[key] = recon.to(CPU_DEVICE)
                        value = value.to(CPU_DEVICE)

                    else:
                        if rank_ratio == 0.0:
                            U = torch.zeros_like(U)
                            s = torch.zeros_like(s)
                            V_T = torch.zeros_like(V_T)
                            parsed_dim = 1
                        elif rank_ratio == 1.0:
                            parsed_dim = dim
                        else:
                            if energy_threshold:
                                singular_energy = s ** 2
                                total_energy = torch.sum(singular_energy)
                                cumulative_energy = torch.cumsum(singular_energy, dim=0)
                                parsed_dim: int = (cumulative_energy / total_energy >= rank_ratio).nonzero(as_tuple=True)[0].min().item() + 1
                            else:
                                parsed_dim: int = max(1, int(rank_ratio * dim))

                        if get_decomposed_tv:
                            svd_vector[key] = {
                                "U": U[:, :parsed_dim].to(CPU_DEVICE),
                                "s": s[:parsed_dim].to(CPU_DEVICE),
                                "V_T": V_T[:parsed_dim, :].to(CPU_DEVICE),
                            }
                        else:
                            recon = (
                                (U[:, :parsed_dim] * s[:parsed_dim]) @ V_T[:parsed_dim, :]
                            )
                            svd_vector[key] = recon.to(CPU_DEVICE)
                        value = value.to(CPU_DEVICE)

                else:
                    if move_origin:
                        if scalar_expansion_list:
                            scalar_expansion_coeff = scalar_expansion_coeff_list[t_idx]
                            svd_vector[key] = scalar_expansion_coeff * (value + prev_origin[key]) - new_origin[key]
                        else:
                            svd_vector[key] = (value + prev_origin[key] - new_origin[key])
                    else:
                        svd_vector[key] = value

                    if rank_ratio == 0.0:
                        svd_vector[key] = torch.zeros_like(svd_vector[key])

            if not get_decomposed_tv:
                _rocon_tv = TaskVector(vector=svd_vector)
            else:
                _rocon_tv = svd_vector

            decomposed_task_vectors.append(_rocon_tv)

        if get_decomposed_tv:
            return decomposed_task_vectors
        else:
            return decomposed_task_vectors

    def _get_origin(self, coeff: float):
        state_dict = deepcopy(self.pretrained_model.state_dict())
        processed_tvec = sum(self.task_vectors)
        for key in state_dict.keys():
            state_dict[key] = state_dict[key] + coeff * processed_tvec.vector[key]
            state_dict[key].to(self.device)
        return state_dict

    def _merge_weights(self, merge_method: str):
        if merge_method == "CART":
            print(f"CART merge method with prior: {self.config.prior} and rank ratio: {self.config.initial_rank_ratio}")
            avg_coeff = 1.0 / len(self.config.tasks)
            _theta_avg = self._get_origin(avg_coeff)
            state_dict = deepcopy(self.pretrained_model.state_dict())

            lowrank_processed_tvec = sum(
                self._get_truncated_task_vectors(
                    rank_ratio=self.config.initial_rank_ratio,
                    prev_origin=state_dict,
                    new_origin=_theta_avg,
                )
            )

            merge_coeff = self.config.prior
            for key in _theta_avg.keys():
                _theta_avg[key] = (
                    _theta_avg[key] + merge_coeff * lowrank_processed_tvec.vector[key]
                )
            
            self._merged_state_dict = _theta_avg

        elif merge_method == "TA":
            print(f"current coefficient: {self.config.prior}")
            self._merged_state_dict = self._get_origin(coeff=self.config.prior)

        elif merge_method == "AVG":
            coeff = 1.0 / len(self.config.tasks)
            print(f"current coefficient: {coeff} with {len(self.config.tasks)} tasks")
            self._merged_state_dict = self._get_origin(coeff)

        else:
            raise ValueError(f"Invalid merge type: {merge_method}")

    def _compress_weight(self, merge_method: str) -> dict[str, Union[ImageEncoder, List[TaskVector]]]:
        ret: dict = {}
        if merge_method == "CART-INDEXING":
            print(f"{merge_method} compression method with rank ratio: {self.config.initial_rank_ratio}")

            avg_coeff = 1.0 / len(self.config.tasks)
            ret["origin"] = self._get_origin(avg_coeff)

            prev_origin = deepcopy(self.pretrained_model.state_dict())
            
            if self.config.get("square_eval", False):
                print("thorough evaluation with respect to rank is enabled")
                ret["tv_list"] = self._get_truncated_task_vectors(
                    rank_ratio=self.config.initial_rank_ratio,
                    prev_origin=prev_origin,
                    new_origin=ret["origin"],
                    specific_singular_val=self.config.specific_singular_val,
                    discrete_rank=self.config.discrete_rank,
                )
            else:
                ret["tv_list"] = self._get_truncated_task_vectors(
                    rank_ratio=self.config.initial_rank_ratio,
                    prev_origin=prev_origin,
                    new_origin=ret["origin"],
                )
        
        else:
            raise ValueError(f"{merge_method} is not implemented.")
        
        return ret

    def get_origin_mat_state_dict(self, merge_coeff: float):
        ret_dict: dict[str, torch.Tensor] = {}
        _get_origin = self._get_origin(merge_coeff)
        for key in _get_origin.keys():
            if len(_get_origin[key].shape) == 2:
                ret_dict[key] = _get_origin[key]
        return ret_dict
    
    def get_mtl_mat_state_dict(self, merge_method: str):
        ret_dict = {}
        
        self._merge_weights(merge_method)
        for key, val in self._merged_state_dict.items():
            if len(val.shape) == 2:
                ret_dict[key] = val
        
        return ret_dict

    def get_image_encoder(self, merge_method: str):
        self._merge_weights(merge_method)
        clone_model = deepcopy(self.pretrained_model)
        clone_model.load_state_dict(self._merged_state_dict)
        return clone_model

    def get_encoder_and_tv(self, merge_method: str) -> dict[str, Union[ImageEncoder, List[TaskVector]]]:
        ret: dict = {}
        _ret = self._compress_weight(merge_method)

        _model = deepcopy(self.pretrained_model)
        _model.load_state_dict(_ret["origin"])
        ret["encoder"] = _model

        ret["tv_list"] = _ret["tv_list"]

        return ret

    def forward(self, x):
        raise NotImplementedError("StaticMergeModule does not support forward method.")

def load_model(config, device, get_raw_weight=False):
    print("Loading models...")
    model_list = {}

    for name in config.tasks:
        TA_mode = is_TA_mode(config, name)
        dir_ret = get_dir_dict(config, TA_mode)
        if TA_mode:
            path_name = name
            finetuned_model_path = os.path.join(dir_ret["weight_root"], path_name, "finetuned.pt")
        else:
            path_name = name + "Val"
            finetuned_model_path = os.path.join(dir_ret["weight_root"], path_name, "nonlinear_finetuned.pt")

        if not os.path.exists(finetuned_model_path):
            raise FileNotFoundError(f"Model file not found: {finetuned_model_path}")
        model = torch.load(finetuned_model_path, map_location=device)
        model_list[name] = model

    zero_shot_encoder = ImageEncoder(args=config, keep_lang=False).to(device)

    if get_raw_weight:
        return {"encoder": zero_shot_encoder, "model_list": model_list}

    print("Constructing task vectors...")
    task_vectors = []
    for task in config.tasks:
        finetuned_state_dict = model_list[task]
        tv = TaskVector(config, zero_shot_encoder.state_dict(), finetuned_state_dict, task=task).to(device)
        task_vectors.append(tv)

    return {
        "encoder": zero_shot_encoder,
        "task_vectors": task_vectors,
    }

def eval(config):
    print(f"eval merge method: {config.merge_method}")

    ret = load_model(config, CPU_DEVICE)
    mtl_model = StaticMergeModule(
        config=config,
        zero_shot_encoder=ret["encoder"],
        task_vectors=ret["task_vectors"],
    )
    
    del ret
    garbage_collect()

    image_encoder = mtl_model.get_image_encoder(config.merge_method)
    test_log = ExperimentResult(
        method=config.merge_method,
        merge_type=config.merge_type,
        num_tasks=len(config.tasks),
        index=getattr(wandb.run, "name", None),
        exp_config=config,
    )
    for task in config.tasks:
        print("Evaluating task: ", task)
        classification_head = get_classification_head(config, task)
        metrics = eval_single_dataset(
            image_encoder=image_encoder.to(config.device),
            classification_head=classification_head.to(config.device),
            dataset_name=task,
            args=config,
        )
        test_log.add_score(task, metrics.get("top1", 0.0), loss=metrics.get("avg_loss", 0.0))

    return test_log

def eval_indexing(config):
    print(f"eval merge method: {config.merge_method}")

    ret = load_model(config, CPU_DEVICE)

    mtl_model = StaticMergeModule(
        config=config,
        zero_shot_encoder=ret["encoder"],
        task_vectors=ret["task_vectors"],
    )
    
    del ret
    garbage_collect()

    model_dict = mtl_model.get_encoder_and_tv(config.merge_method)

    test_log = ExperimentResult(
        method=config.merge_method,
        merge_type=config.merge_type,
        num_tasks=len(config.tasks),
        index=getattr(wandb.run, "name", None),
        exp_config=config,
    )
    for idx, task in enumerate(config.tasks):
        _model: ImageEncoder = deepcopy(model_dict["encoder"])
        tv: TaskVector = model_dict["tv_list"][idx]
        classification_head = get_classification_head(config, task)

        print(f"Evaluating task: {task} with coefficient: {config.prior}")
        tv.apply_to(_model, scaling_coef=config.prior)

        metrics = eval_single_dataset(
            image_encoder=_model.to(config.device),
            classification_head=classification_head.to(config.device),
            dataset_name=task,
            args=config,
        )
        test_log.add_score(task, metrics.get("top1", 0.0), loss=metrics.get("avg_loss", 0.0))
    return test_log

def each_config_run(config_path):
    config = load_config(config_path)
    device = config.device
    
    if config.merge_type == "static":
        print("Static Merging Module")
        ret = eval(config)

    elif config.merge_type == "indexing":
        ret = eval_indexing(config)

    result_instance = ret
    result = result_instance.to_dict()
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpu_split",
        type=int,
        default=1
    )

    parser.add_argument(
        "--config_list_path",
        type=str,
        required=True,
        help="Path to the config list file.",
    )
    args = parser.parse_args()

    print("single run mode")
    each_config_run(args.config_list_path)