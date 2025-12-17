CUDA_VISIBLE_DEVICES=1 python train.py \
 -s data/robustnerf/yoda \
 --port 6019 \
 --expname yoda/semantic \
 --configs arguments/image_dataset/robustnerf.py \
