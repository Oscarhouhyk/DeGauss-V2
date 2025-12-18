CUDA_VISIBLE_DEVICES=1 python render_gaussian_dynerf.py \
 -s data/dynerf/cut_roasted_beef \
 --port 6017 \
 --expname Neu3Drender_cutmirror \
 --configs arguments/video_dataset/Neu3D.py \
 --render_checkpoint output/cut_roasted_beef_cutmirror \
