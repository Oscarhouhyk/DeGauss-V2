CUDA_VISIBLE_DEVICES=1 python render_gaussian_dynerf.py \
 -s /data/hejq/DeGauss/data/hot3d/aria_seq_1 \
 --port 6017 \
 --expname hot3d_semantic \
 --configs arguments/image_dataset/robustnerf.py \
 --render_checkpoint output/yoda/semantic \
