# requirements
- please install requried packages
```bash
pip install -r requirement.txt
```
# Finetuned weight
- please place finetuend weights to `/workspace/weight/` following https://github.com/nik-dim/tall_masks

# Datasets
- please place propely processed datasets to `/workspace/data/` following https://github.com/nik-dim/tall_masks

# run script
```bash
# 8 tasks with CLIP ViT-B/32
python src/merging.py  --config_list_path ./configs/CART/ViT-b-32_8_CART.yaml
# 14 tasks with CLIP ViT-B/32
python src/merging.py  --config_list_path ./configs/CART/ViT-b-32_14_CART.yaml
# 20 tasks with CLIP ViT-B/32
python src/merging.py  --config_list_path ./configs/CART/ViT-b-32_20_CART.yaml

```
