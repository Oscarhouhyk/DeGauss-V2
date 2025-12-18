CUDA_VISIBLE_DEVICES=7 python train.py \
 -s /data/houyj/multimodal/DeGauss/data/nerf-on-the-go/mountain \
 --port 6018 \
 --expname mountain_semantic \
 --configs arguments/image_dataset/nerfonthego_outdoor.py \
