import torch
from typing import Dict

def make_task_vector_low_rank(args, finetuned_single_weight, pretrained_single_weight, key):
    """Create a low rank task vector for a single weight tensor."""
    diff = finetuned_single_weight - pretrained_single_weight

    return_diff_keys = {'model.logit_scale'}
    diff_substrings = ['ln', 'bias']
    low_rank_substrings = ['attn', 'mlp']

    if key in return_diff_keys or any(sub in key for sub in diff_substrings):
        return diff

    elif any(sub in key for sub in low_rank_substrings):
        U, s, V_T = torch.linalg.svd(diff.to(args.device), full_matrices=False)
        dim = s.shape[0]
        parsed_dim = max(1, int(args.initial_rank_ratio * dim))

        sqrt_s = torch.sqrt(s[:parsed_dim])
        parsed_V_T = torch.diag(sqrt_s) @ V_T[:parsed_dim, :]
        parsed_U = U[:, :parsed_dim] @ torch.diag(sqrt_s)

        recon = (parsed_U @ parsed_V_T).to("cpu")
        return recon

    else:
        return diff


class TaskVector():
    def __init__(self, args=None, pretrained_checkpoint=None, finetuned_checkpoint=None, task=None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.

        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
            if getattr(args, "do_auto_rank", False):
                self.vector_rank_dict: Dict[str, int] = {}
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                pretrained_state_dict = pretrained_checkpoint
                finetuned_state_dict = finetuned_checkpoint
                print(f'{task} Building task vector')
                self.vector = {}
                
                Warning(f"low rank approximation in TaskVector class is deprecated. Please manually do SVD approximation")
                for key in pretrained_state_dict:
                #     if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                #         continue
                #     assert args.initial_rank_ratio >= 0.0 and args.initial_rank_ratio <= 1.0, "initial_rank_ratio should be in [0, 1]"
                #     if args.initial_rank_ratio < 1.0:
                #         self.vector[key] = make_task_vector_low_rank(
                #             args, finetuned_state_dict[key], pretrained_state_dict[key], key)
                #     else:
                #         self.vector[key] = finetuned_state_dict[key] - \
                #             pretrained_state_dict[key]
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]

    def to(self, device):
        for key in self.vector:
            self.vector[key] = self.vector[key].to(device)
        return self

    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            all_keys = set(self.vector.keys()).union(set(other.vector.keys()))
            for key in all_keys:
                if key in self.vector and key in other.vector:
                    # 같은 키는 합침
                    new_vector[key] = self.vector[key] + other.vector[key]
                elif key in self.vector:
                    # self.vector에만 있는 키
                    new_vector[key] = self.vector[key]
                else:
                    # other.vector에만 있는 키
                    new_vector[key] = other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def multiply(self, scalar):
        """Multiply a task vector by a scalar."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = scalar * self.vector[key]
        return TaskVector(vector=new_vector)
    
    def __mul__(self, scalar):
        return self.multiply(scalar)
    
    def __rmul__(self, scalar):
        return self.multiply(scalar)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            if isinstance(pretrained_checkpoint, str):
                pretrained_model = torch.load(pretrained_checkpoint)
            elif isinstance(pretrained_checkpoint, torch.nn.Module):
                pretrained_model = pretrained_checkpoint
            else:
                raise ValueError(
                    "pretrained_checkpoint must be a file path or a model")
            state_dict = pretrained_model.state_dict()
            for key in state_dict:
                if key in self.vector:
                    state_dict[key] = state_dict[key] + \
                        scaling_coef * self.vector[key]
                else:
                    print(
                        f"Key {key} not found in task vector. Copying from pretrained model.")
                    state_dict[key] = state_dict[key]
            for key in self.vector:
                if key not in state_dict:
                    print(
                        f"Key {key} found only in task vector. Adding to new state dict.")
                    state_dict[key] = self.vector[key]
            pretrained_model.load_state_dict(state_dict, strict=True)
            return pretrained_model
