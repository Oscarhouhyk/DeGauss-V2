# 2025 Fall 多模态学习 Final Project：DeGauss-V2
项目基于 2025 ICCV Paper [DeGauss](https://github.com/BatFaceWayne/DeGauss)
小组成员：何嘉齐 侯玉杰 2300013108

## Environmental Setups

```bash
git clone https://github.com/Oscarhouhyk/DeGauss-V2.git
cd DeGauss-V2
git submodule update --init --recursive
```
We recommend you to set up the environment via conda.
```bash
conda create -n DeGaussV2 python=3.11
conda activate DeGaussV2

pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

## 数据集链接 & 数据集结构
**我们在以下的数据集中各挑选了一个场景进行项目复现以及实验。**
1. Egocentric video sequences: ADT, AEA, Hot3D , Epic-Field
For Aria Datasets, please refer to [project-aria](https://www.projectaria.com/resources/#resources-datasets) and prepare data with [Nerfstudio](https://docs.nerf.studio/quickstart/custom_dataset.html#aria). The EPIC-Field dataset could be accessed [here](https://epic-kitchens.github.io/epic-fields/).  
2. [NerF On-the-Go and RobustNerf](https://borealisdata.ca/dataset.xhtml?persistentId=doi%3A10.5683%2FSP3%2FWOFXFT), please follow the instructions in [SpotLessSplats](https://github.com/lilygoli/SpotLessSplats/tree/main) for dataset processing.
3. [Neu3D](https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0), Please follow the instructions in [4DGaussians](https://github.com/hustvl/4DGaussians) for preparation. You could find processed fused.ply file of Neu3D dataset [here](https://drive.google.com/file/d/1oTtwku3ITuijdMxcw6QOcsNSA6aMOqFX/view). 
4. [HyperNeRF](https://github.com/google/hypernerf/releases/tag/v0.1)

Please refer to the original code [DeGauss](https://github.com/BatFaceWayne/DeGauss) for Data Preprocessing methods. 

The dataset structure should look follows

```
├── data
│   | hypernerf
│     ├── interp
│     ├── misc
│     ├── virg
│   | dynerf
│     ├── cook_spinach
│       ├── cam00
│           ├── images
│               ├── 0000.png
│               ├── 0001.png
│               ├── 0002.png
│               ├── ...
│     ...
|  | Nerf on-the-go/RobustNerf
|     ├── fountain
│       ├── images
│           ├── 1extra000.JPG
|           ...
│           ├── 2clutter008.JPG
|           ...
|     ├── mountain
|     ...
|  | Aria Digital Twin /Aria Everyday Activities / Hot3D
|     ├── Seq1
│       ├── images
│       ├── masks
│       |__ global_points.ply
|     ...

```

To visualize gaussian models, we recommend using this amazing gaussian splatting visualizing tool: [online visualizer](https://antimatter15.com/splat/).

## Training--原论文复现
For training video datasets as  `cut_roasted_beef` of Neu3D dataset, run
```python
##### please refer to the configs in folder arguments for different dataset setup
python train.py -s data/dynerf/cut_roasted_beef --port 6019 --expname cut_roasted_beef --configs arguments/video_dataset/Neu3D.py
```

For training image datasets for distractor-free scene modeling as Neu3D scenes such as `cut_roasted_beef`, run
```python
######## please use configs nerfonthego.py for indoor scenes
python3 train.py -s data/nerf-on-the-go/mountain --port 6019 --expname mountain --configs arguments/image_dataset/nerfonthego_outdoor.py
```

## 运行改进后的实验


## Rendering

Run the following script to render Neu3D dataset.
```python
######## please use configs nerfonthego.py for indoor scenes
python render_gaussian_dynerf.py -s path_to_dataset --port 6017 --expname Neu3Drender --configs
arguments/video_dataset/Neu3D.py" --render_checkpoint path_to_checkpoint
```

## Evaluation

You can just run the following script to evaluate the model.

```python
#### for dynamic scene eval -d : output base folder -s scene name
python calc_metric.py -d './test/' -s flame_steak_sparse

#### for distractor static scene eval -d : output base folder -s scene name
python calc_metric_static.py -d './test/' -s patio_high 

```



