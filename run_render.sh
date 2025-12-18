CUDA_VISIBLE_DEVICES=1 python render_gaussian_dynerf.py \
 -s data/robustnerf/yoda \
 --port 6017 \
 --expname RobustNerfrender_semantic \
 --configs arguments/image_dataset/robustnerf.py \
 --render_checkpoint output/yoda/semantic \
