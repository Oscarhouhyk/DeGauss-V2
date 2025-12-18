import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import cv2
import torch
import argparse
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Modify as needed

class MirrorDetector:
    def __init__(self, model_name="nvidia/segformer-b4-finetuned-ade-512-512", device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Initializing MirrorDetector with model: {model_name} on {self.device}")
        
        try:
            self.processor = SegformerImageProcessor.from_pretrained(model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please ensure you have internet access to download the model or provide a local path.")
            self.model = None

        # ADE20K Class Indices
        # 8: windowpane;window
        # 28: mirror
        self.target_classes = [8, 28]

    def predict(self, image_path):
        """
        Run inference on a single image.
        Returns a binary mask (numpy array, 0-255) where 255 is mirror or window.
        """
        if self.model is None:
            return None

        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            original_size = image.size # (W, H)
            
            # Preprocess
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

            # Upsample logits to original image size
            # Note: SegFormer outputs are 1/4 resolution, need to upsample
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=(original_size[1], original_size[0]), # (H, W)
                mode="bilinear",
                align_corners=False,
            )

            # Get prediction map
            pred_seg = upsampled_logits.argmax(dim=1)[0] # (H, W)
            
            # Create binary mask for target classes
            mask = torch.zeros_like(pred_seg, dtype=torch.uint8)
            for cls_idx in self.target_classes:
                mask = torch.where(pred_seg == cls_idx, torch.tensor(255, dtype=torch.uint8).to(self.device), mask)
                
            return mask.cpu().numpy()
        except Exception as e:
            print(f"\nError processing {image_path}: {e}")
            return None

def process_folder(detector, images_dir, output_dir):
    if not os.path.exists(images_dir):
        return

    os.makedirs(output_dir, exist_ok=True)
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"Processing {len(image_files)} images in {images_dir}...")

    for img_file in tqdm(image_files):
        img_path = os.path.join(images_dir, img_file)
        save_path = os.path.join(output_dir, os.path.splitext(img_file)[0] + ".png")
        
        # Skip if already exists (optional)
        # if os.path.exists(save_path): continue

        mask = detector.predict(img_path)
        
        if mask is not None:
            # Save mask
            cv2.imwrite(save_path, mask)

def process_dataset(dataset_path, model_name):
    """
    Walks through the dataset, finds images, generates masks, and saves them.
    Supports:
    1. Standard structure: dataset_path/images/*.png
    2. Multi-camera structure: dataset_path/cam*/images/*.png
    """
    
    detector = MirrorDetector(model_name=model_name)
    if detector.model is None:
        return

    # Check for standard structure
    if os.path.exists(os.path.join(dataset_path, "images")):
        print("Detected standard dataset structure.")
        process_folder(detector, os.path.join(dataset_path, "images"), os.path.join(dataset_path, "mirror_masks"))
    
    # Check for multi-camera structure (cam00, cam01, ...)
    subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    cam_dirs = [d for d in subdirs if d.startswith("cam")]
    
    if cam_dirs:
        print(f"Detected multi-camera dataset structure with {len(cam_dirs)} cameras.")
        for cam_dir in cam_dirs:
            cam_path = os.path.join(dataset_path, cam_dir)
            images_dir = os.path.join(cam_path, "images")
            output_dir = os.path.join(cam_path, "mirror_masks")
            
            if os.path.exists(images_dir):
                process_folder(detector, images_dir, output_dir)
            else:
                print(f"Warning: No 'images' folder found in {cam_dir}")

    if not os.path.exists(os.path.join(dataset_path, "images")) and not cam_dirs:
        print(f"Error: Could not find 'images' folder or 'cam*' folders in {dataset_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mirror/window masks using SegFormer (ADE20K).")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset root")
    parser.add_argument("--model_name", type=str, default="nvidia/segformer-b4-finetuned-ade-512-512", help="HuggingFace model name or path")
    
    args = parser.parse_args()
    
    process_dataset(args.dataset_path, args.model_name)
