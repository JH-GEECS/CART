# Revisiting Weight Averaging for Model Merging

TL;DR:  By centering task vectors and applying low-rank approximations, our method (CART) effectively merges fine-tuned models for multi-task learning, significantly reducing inter-task interference and nearly matching traditional multi-task performance.

Multi-Task Performances on 8, 14 and 20 vision tasks with merged ViT-B/32 and ViT-L/14.
| **Method**                      | **ViT-B/32** |              |              | **ViT-L/14** |              |              |
|---------------------------------|--------------|--------------|--------------|--------------|--------------|--------------|
|                                 | **8 tasks**  | **14 tasks** | **20 tasks** | **8 tasks**  | **14 tasks** | **20 tasks** |
| Pretrained                      | 48.0         | 57.1         | 56.0         | 65.0         | 68.4         | 65.4         |
| Individual                      | 90.5         | 89.6         | 90.5         | 94.2         | 93.3         | 94.1         |
| **Without Test Time Adaptation**|              |              |              |              |              |              |
| Weight Averaging                | 65.9         | 64.3         | 61.0         | 79.6         | 76.8         | 71.7         |
| Task Arithmetic                 | 69.1         | 65.4         | 60.5         | 84.5         | 79.6         | 74.2         |
| Ties-Merging                    | 72.4         | 65.2         | 62.9         | 86.1         | 79.5         | 75.8         |
| Consensus TA                    | 75.2         | 70.0         | 65.0         | 86.6         | 81.9         | 78.8         |
| **CART (Ours)**                 | **84.7**     | **79.5**     | **76.3**     | **92.6**     | **88.7**     | **87.9**     |
| **With Test Time Adaptation**   |              |              |              |              |              |              |
| Adamerging+TA                   | 80.1         | 76.7         | 69.2         | 90.8         | 88.0         | 86.8         |
| **Adamerging+CART (Ours)**      | **85.8**     | **82.3**     | **82.7**     | **93.1**     | **90.4**     | **91.3**     |

paper link: [Arxiv](https://arxiv.org/abs/2412.12153)

## Requirements

Please install the required packages:
```bash
pip install -r requirements.txt
```

## Finetuned Weights

Please download the finetuned weights from [JH-C-k/ViT-B-32](https://huggingface.co/JH-C-k/ViT-B-32/tree/main) and place them in `/workspace/weight/checkpoints`.

Alternatively, you can place the finetuned weights in `/workspace/weight/`, following the structure from [TALL-Masks](https://github.com/nik-dim/tall_masks).

- For 8 vision tasks, we used the finetuned models provided in [Editing Models with Task Arithmetic](https://github.com/mlfoundations/task_vectors).
- For addition 12 tasks in 14 and 20 vision tasks, we used the finetuned models from [TALL-Masks](https://github.com/nik-dim/tall_masks).
- Therefore, the evaluation results may vary if you following the instruction of [TALL-Masks](https://github.com/nik-dim/tall_masks).

## Datasets

Please download the datasets from [JH-C-k/Merge_dataset](https://huggingface.co/datasets/JH-C-k/Merge_dataset) and place them in `/workspace/data/`.

Then, follow the instructions below to unzip the datasets:
```bash
apt-get install pigz
cat /downloaded_path/* | pigz -p 32 -d -c | tar -xvf - -C /workspace/data/
```

Since the dataset is approximately 100GB for 20 vision tasks, it is recommended to use `hf_transfer` to download the dataset. For detailed instructions, refer to the [Hugging Face Hub guide](https://huggingface.co/docs/huggingface_hub/guides/download).

Alternatively, you can place properly processed datasets in `/workspace/data/`, following the structure from [TALL-Masks](https://github.com/nik-dim/tall_masks).

- Note: The data split may differ, as reported in [this issue](https://github.com/nik-dim/tall_masks/issues/1). Therefore, evaluation results may vary.

## Run Script

```bash
# 8 tasks with CLIP ViT-B/32
python src/merging.py --config_list_path ./configs/CART/ViT-b-32_8_CART.yaml

# 14 tasks with CLIP ViT-B/32
python src/merging.py --config_list_path ./configs/CART/ViT-b-32_14_CART.yaml

# 20 tasks with CLIP ViT-B/32
python src/merging.py --config_list_path ./configs/CART/ViT-b-32_20_CART.yaml
```


## citation
```bibtex
@article{choi2024revisiting,
  title={Revisiting weight averaging for model merging},
  author={Choi, Jiho and Kim, Donggyun and Lee, Chanhyuk and Hong, Seunghoon},
  journal={arXiv preprint arXiv:2412.12153},
  year={2024}
}
```
