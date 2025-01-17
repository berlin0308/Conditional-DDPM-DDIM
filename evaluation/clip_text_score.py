import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple
import os

def calculate_clip_text_scores_folder(folder_path: str, text: str) -> List[Tuple[str, float]]:
    # Load the CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Get all image files from the folder
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    image_paths = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.lower().endswith(image_extensions)
    ]

    # Check if the number of images is equal to 100
    if len(image_paths) != 25:
        raise ValueError(f"The number of images in the folder must be exactly 25. Found {len(image_paths)} images.")

    # Process images in batches to avoid memory issues
    batch_size = 32
    all_scores = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        
        # Load and preprocess the images
        images = [Image.open(path).convert('RGB') for path in batch_paths]
        inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)

        # Calculate the CLIP scores
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            clip_scores = logits_per_image.squeeze().tolist()

        # Handle single image case
        if isinstance(clip_scores, float):
            clip_scores = [clip_scores]

        # Pair each image path with its score
        all_scores.extend(clip_scores)

    return all_scores