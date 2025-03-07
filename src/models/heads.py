import os
import torch
from tqdm import tqdm

import open_clip

from src.dataset.templates import get_templates
from src.dataset.registry import get_dataset

from src.models.modeling import ClassificationHead, ImageEncoder
from src.utils import is_TA_mode, get_dir_dict


def build_classification_head(model, dataset_name, template, data_location, device):
    template = get_templates(dataset_name)

    logit_scale = model.logit_scale
    dataset = get_dataset(
        dataset_name,
        None,
        location=data_location
    )
    model.eval()
    model.to(device)

    print('Building classification head.')
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(dataset.classnames):
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = open_clip.tokenize(texts).to(device)  # tokenize
            embeddings = model.encode_text(texts)  # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        # output: [num_classes, 1, embedding_size]
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()

        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    classification_head = ClassificationHead(
        normalize=True, weights=zeroshot_weights)

    return classification_head


def get_classification_head(config, dataset):
    TA_mode = is_TA_mode(config, dataset)
    dir_ret = get_dir_dict(config, TA_mode)
    if TA_mode:
        filename = os.path.join(
            dir_ret["save"], f'head_{dataset}.pt')
        state_dict = torch.load(filename)
    else:
        filename = os.path.join(
            dir_ret["save"], f'head_{dataset}Val.pt')
    if os.path.exists(filename):
        print(
            f'Classification head for {config.model} on {dataset} exists at {filename}')
        if TA_mode:
            cls_head = ClassificationHead(
                normalize=True, weights=state_dict['weight'])
            return cls_head       
        else:
            return ClassificationHead.load(filename)
    print(
        f'Did not find classification head for {config.model} on {dataset} at {filename}, building one from scratch.')
    model = ImageEncoder(config, keep_lang=True).model
    template = get_templates(dataset)
    classification_head = build_classification_head(
        model, dataset, template, config.data_location, config.device)
    os.makedirs(config.save, exist_ok=True)
    classification_head.save(filename)
    return classification_head
