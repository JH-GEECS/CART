from dataclasses import dataclass, field
from typing import Dict, Optional

import os
import pandas as pd
import namesgenerator
import yaml

from typing import Optional, List, Union

from src.dataset.registry import TASK_MAP


LOGGING_TARGET_HYPS = [
    "index",
    "model",
    "merge_type",
    "merge_method",
    "num_tasks",
    "initial_rank_ratio",
    "prior",
    "initial_merge_coeff",
    "initial_merge_rank_ratio",
    "external_weight_path"
]


@dataclass
class ExperimentResult:
    method: str
    merge_type: str
    num_tasks: int

    control_tasks: Optional[List[str]] = None
    scores: Union[Dict[str, float], Dict[str, Dict[str, float]]] = None
    avg_losses: Union[Dict[str, float], Dict[str, Dict[str, float]]] = None
    index: Optional[str] = None
    exp_config: Optional[Dict] = None

    def __post_init__(self):
        if self.index is None:
            self.index = namesgenerator.get_random_name()

        if self.exp_config is not None:
            self.exp_config.update({"index": self.index})
            self.exp_config.update({"num_tasks": self.num_tasks})

        assert self.num_tasks in [
            8, 14, 20], f"num_tasks should be one of [8, 14, 20], but got {num_tasks}"

        # define score table
        if self.control_tasks is not None:
            self.scores = {
                target: {control_task: 0.0 for control_task in self.control_tasks}
                for target in TASK_MAP[self.num_tasks]
            }
            self.avg_losses = {
                target: {control_task: 0.0 for control_task in self.control_tasks}
                for target in TASK_MAP[self.num_tasks]
            }
        else:
            self.scores = {}
            self.avg_losses = {}
            for task in TASK_MAP[self.num_tasks]:
                self.scores[task] = 0.0
                self.avg_losses[task] = 0.0

    def add_score(self, task: str, score: float, loss: Optional[float] = None, control_task: Optional[str] = None,):
        assert task in self.scores.keys(
        ), f"{task} is not in self.scores.keys()"
        if self.control_tasks is not None:
            assert control_task is not None, "control task should be provided."
            self.scores[task][control_task] = score

            if loss is not None:
                self.avg_losses[task][control_task] = loss

        else:
            self.scores[task] = score

            if loss is not None:
                self.avg_losses[task] = loss

    def get_score(self, task: str, control: Optional[str] = None):
        if self.control_tasks is not None:
            if control is not None:
                return self.scores.get(task, {}).get(control, None)
            else:
                return self.scores.get(task, None)  # target_task에 대해서 전체 반환
        else:
            return self.scores.get(task, None)

    def _get_avg_score(self):
        if self.control_tasks is not None:
            return None
        else:
            avg_score = sum(self.scores.values()) / len(self.scores)
            return avg_score

    def to_dict(self):
        if self.exp_config is not None:
            base = {}
            for key in self.exp_config.keys():
                if key in LOGGING_TARGET_HYPS:
                    _value = self.exp_config.get(key, None)
                    if _value is not None:
                        base[key] = str(_value)
                    else:
                        pass
        else:
            base = {
                "index": self.index,
                "num_tasks": self.num_tasks,
                "method": self.method,
                "merge_type": self.merge_type,
            }
        result_columns = ["scores"]
        if getattr(self, "avg_losses", None) is not None:
            result_columns.append("avg_losses")

        for each_result in result_columns:
            if self.control_tasks is not None:
                base[each_result] = {}
                for task in getattr(self, each_result).keys():
                    base[each_result][task] = getattr(self, each_result)[task]
            else:
                for task in getattr(self, each_result).keys():
                    if each_result == "scores":
                        base[task] = getattr(self, each_result)[task]
                    else:
                        base[f"{each_result}_{task}"] = getattr(self, each_result)[task]

                if each_result == "scores":
                    base["avg"] = self._get_avg_score()


        return base

