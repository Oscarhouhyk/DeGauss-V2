import torch
import torchvision.transforms as T
from PIL import Image
import os
import numpy as np

class SemanticFeatureExtractor:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self.transform = T.Compose([
            T.Resize((518, 518)), # DINOv2 usually works well with 518x518 or multiples of 14
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def load_model(self):
        if self.model is None:
            print("Loading DINOv2 model...")
            try:
                self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
                self.model.to(self.device)
                self.model.eval()
                print("DINOv2 model loaded.")
            except Exception as e:
                print(f"Failed to load DINOv2 model: {e}")
                self.model = None

    def extract_features(self, image_path):
        if self.model is None:
            self.load_model()
        
        if self.model is None:
            return None

        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features_dict = self.model.forward_features(img_tensor)
                features = features_dict['x_norm_patchtokens']
                
            return features.cpu()
        except Exception as e:
            print(f"Error extracting features for {image_path}: {e}")
            return None

    def get_semantic_mask(self, features, image_shape, dynamic_classes=None):
        # This is a placeholder. In a real scenario, you would map features to classes.
        # For DINOv2, features are not directly classes. 
        # You might need a segmentation head or use PCA/clustering to find dynamic objects.
        # Given the user request, we assume we can identify "dynamic" regions.
        # If we don't have a segmentation head, we might rely on the user providing masks 
        # or use a simple heuristic if possible, but DINOv2 features are high-dimensional.
        
        # If the user meant "Semantic Segmentation" (like SegFormer or Mask2Former), 
        # then DINOv2 is just a backbone.
        # However, the prompt says "Features from a generalized visual encoder (like DINOV2)".
        # And "Gaussians whose locations correspond to semantic classes known to be dynamic".
        
        # Assuming we have some way to map features to dynamic probability.
        # For now, we will return a dummy mask or try to use a simple method if possible.
        # But without a trained decoder, we can't get classes from DINOv2 directly.
        
        # Maybe the user implies using DINOv2 features for similarity or clustering?
        # Or maybe they want us to use a semantic segmentation model INSTEAD of just DINOv2 features?
        # "Gaussians whose locations correspond to semantic classes known to be dynamic" strongly implies class labels.
        
        # I will implement a dummy "is_dynamic" check based on features if possible, 
        # or better, I will use a pre-trained semantic segmentation model if available in torchvision.
        # torchvision has DeepLabV3.
        pass

class SemanticSegmentation:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self.transform = T.Compose([
            T.Resize(520),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        # COCO classes where person is 15 (if using DeepLabV3 ResNet50)
        # Pascal VOC classes: person is 15.
        # Let's check torchvision DeepLabV3 classes.
        # Expanded dynamic classes: aeroplane, bicycle, bird, boat, bus, car, cat, cow, dog, horse, motorbike, person, sheep, train
        self.dynamic_classes = [1, 2, 3, 4, 6, 7, 8, 10, 12, 13, 14, 15, 17, 19]

    def load_model(self):
        if self.model is None:
            print("Loading DeepLabV3 model...")
            try:
                self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
                self.model.to(self.device)
                self.model.eval()
                print("DeepLabV3 model loaded.")
            except Exception as e:
                print(f"Failed to load DeepLabV3 model: {e}")
                self.model = None

    def get_dynamic_mask(self, image_path, target_size):
        if self.model is None:
            self.load_model()
            
        if self.model is None:
            return torch.zeros(target_size, device=self.device)

        try:
            img = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)['out'][0]
            
            output_predictions = output.argmax(0)
            
            # Create mask for dynamic classes
            mask = torch.zeros_like(output_predictions, dtype=torch.float32)
            for cls in self.dynamic_classes:
                mask[output_predictions == cls] = 1.0
                
            # Resize to target size
            mask = mask.unsqueeze(0).unsqueeze(0) # 1, 1, H, W
            mask = torch.nn.functional.interpolate(mask, size=target_size, mode='nearest')
            
            return mask.squeeze()
        except Exception as e:
            print(f"Error getting semantic mask for {image_path}: {e}")
            return torch.zeros(target_size, device=self.device)

