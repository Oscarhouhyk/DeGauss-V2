# 2025 Fall 多模态学习 Final Project：DeGauss-V2
项目基于 2025 ICCV Paper [DeGauss](https://github.com/BatFaceWayne/DeGauss)

小组成员：何嘉齐 2300013184 侯玉杰 2300013108

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

## 数据集
**我们在以下的数据集中各挑选了一个场景进行项目复现以及实验。**
1. Aria Datesets: ADT, AEA, Hot3D, Epic-Field. For Aria Datasets, please refer to [project-aria](https://www.projectaria.com/resources/#resources-datasets) and prepare data with [Nerfstudio](https://docs.nerf.studio/quickstart/custom_dataset.html#aria). The EPIC-Field dataset could be accessed [here](https://epic-kitchens.github.io/epic-fields/).  

2. [NerF On-the-Go and RobustNerf](https://borealisdata.ca/dataset.xhtml?persistentId=doi%3A10.5683%2FSP3%2FWOFXFT), please follow the instructions in [SpotLessSplats](https://github.com/lilygoli/SpotLessSplats/tree/main) for dataset processing.

3. [Neu3D](https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0), Please follow the instructions in [4DGaussians](https://github.com/hustvl/4DGaussians) for preparation. You could find processed fused.ply file of Neu3D dataset [here](https://drive.google.com/file/d/1oTtwku3ITuijdMxcw6QOcsNSA6aMOqFX/view). 

4. [HyperNeRF](https://github.com/google/hypernerf/releases/tag/v0.1)

Please refer to the original codebase [DeGauss](https://github.com/BatFaceWayne/DeGauss) for Data Preprocessing methods and dataset structure. 

To visualize gaussian models, we recommend using this amazing gaussian splatting visualizing tool: [online visualizer](https://antimatter15.com/splat/).


## Training--原论文复现
Run the following training script, please refer to the DeGauss codebase for changing the configs and the corresponding folder arguments for different dataset setup
```bash
bash run_train.sh
```

## Rendering
Run the following script to render the scene after training.
```bash
######## please refer to the configs in the rendering script
bash run_render.sh
```

## Evaluation
You can just run the following script to evaluate the model.
```bash
#### change the corresponding configs/scripts for dynamic/static scene
bash run_evaluation.sh
```

## 运行改进后的实验
1. 解决 semi-static objects 问题
```bash
git checkout semantic
```

2. 解决镜面反射问题
```bash
git checkout illu
```

**training/evaluation/render 运行方式同上，更改文件夹输出名称即可**



