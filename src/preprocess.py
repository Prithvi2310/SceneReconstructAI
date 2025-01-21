import os
import json
import cv2
import numpy as np
from tqdm import tqdm

# Paths
COCO_IMAGES_DIR = "data/coco/images"
COCO_ANNOTATIONS_DIR = "data/coco/annotations"
PREPROCESSED_DIR = "data/coco/preprocessed"

def load_annotations(annotations_path):
    """
    Load COCO annotations from JSON.
    """
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    return annotations

def preprocess_image(image_path):
    """
    Preprocess a single image: Load, normalize.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    
    # Normalize image (scale pixel values to [0, 1])
    img = img / 255.0  # Convert to float32 in range [0, 1]
    
    return img

def preprocess_dataset(split):
    """
    Preprocess images and annotations for a specific split (train, val, test).
    """
    # Set paths
    images_dir = os.path.join(COCO_IMAGES_DIR, f"{split}2017")
    annotations_path = os.path.join(COCO_ANNOTATIONS_DIR, f"instances_{split}2017.json")
    
    # Load annotations
    annotations = load_annotations(annotations_path)
    
    # Create output directories
    output_images_dir = os.path.join(PREPROCESSED_DIR, f"{split}2017")
    os.makedirs(output_images_dir, exist_ok=True)
    
    print(f"Processing {split} dataset...")
    for image_info in tqdm(annotations['images'], desc=f"Processing {split} images"):
        image_id = image_info['id']
        file_name = image_info['file_name']
        image_path = os.path.join(images_dir, file_name)
        
        try:
            # Preprocess image
            preprocessed_image = preprocess_image(image_path)
            
            # Save preprocessed image
            save_path = os.path.join(output_images_dir, file_name)
            cv2.imwrite(save_path, (preprocessed_image * 255).astype(np.uint8))  # Save as uint8
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    
    print(f"{split} preprocessing complete! Processed images saved to {output_images_dir}")

if __name__ == "__main__":
    # Ensure the preprocessed directory exists
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    
    # Preprocess each dataset split
    for split in ["train", "val"]:
        preprocess_dataset(split)
