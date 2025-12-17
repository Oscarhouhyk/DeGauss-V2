from utils.semantic_utils import SemanticSegmentation
from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov
class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type
        self.semantic_segmenter = SemanticSegmentation()

    # def __getitem__(self, index):
    #     # breakpoint()
    #
    #     if self.dataset_type != "PanopticSports":
    #         try:
    #             image, w2c, time = self.dataset[index]
    #             R, T = w2c
    #             FovX = focal2fov(self.dataset.focal[0], image.shape[2])
    #             FovY = focal2fov(self.dataset.focal[0], image.shape[1])
    #             mask = None
    #         except:
    #             caminfo = self.dataset[index]
    #             image = caminfo.image
    #             R = caminfo.R
    #             T = caminfo.T
    #             FovX = caminfo.FovX
    #             FovY = caminfo.FovY
    #             time = caminfo.time
    #
    #             mask = caminfo.mask
    #         # return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
    #         #                   image_name=f"{index}",uid=index,data_device=torch.device("cuda"),time=time,
    #         #                   mask=mask, SD_feature = torch.from_numpy(caminfo.SD_feature).float().squeeze(-1))
    #         return Camera(colmap_id=index, R=R, T=T, FoVx=FovX, FoVy=FovY, image=image, gt_alpha_mask=None,
    #                       image_name=f"{index}", uid=index, data_device=torch.device("cuda"), time=time,
    #                       mask=mask, SD_feature=1, semantic_mask=semantic_mask)
    #     else:
    #         return self.dataset[index]

    def __getitem__(self, index):
        # breakpoint()

        if self.dataset_type != "PanopticSports":
            try:
                image, w2c, time = self.dataset[index]
                R,T = w2c
                FovX = focal2fov(self.dataset.focal[0], image.shape[2])
                FovY = focal2fov(self.dataset.focal[0], image.shape[1])
                mask=None
            except:
                caminfo = self.dataset[index]
                image = caminfo.image
                R = caminfo.R
                T = caminfo.T
                FovX = caminfo.FovX
                FovY = caminfo.FovY
                time = caminfo.time

                mask = caminfo.mask

            semantic_mask = None
            try:
                if hasattr(caminfo, 'image_path'):
                    target_size = (image.shape[1], image.shape[2])
                    semantic_mask = self.semantic_segmenter.get_dynamic_mask(caminfo.image_path, target_size)
            except:
                pass
            try:
                temp_name = caminfo.image_name
            except:
                return Camera(colmap_id=index, R=R, T=T, FoVx=FovX, FoVy=FovY, image=image, gt_alpha_mask=None,
                                      image_name=f"{index}", uid=index, data_device=torch.device("cuda"), time=time,
                                      mask=mask, SD_feature=1, semantic_mask=semantic_mask)
            return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
                              image_name=caminfo.image_name,uid=index,data_device=torch.device("cuda"),time=time,
                              mask=mask, semantic_mask=semantic_mask)
        else:
            return self.dataset[index]
    def __len__(self):
        
        return len(self.dataset)
