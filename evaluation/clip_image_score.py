import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple
import os
from collections import defaultdict

def load_images_from_folder(folder_path: str) -> List[Tuple[str, Image.Image]]:
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(image_extensions):
            file_path = os.path.join(folder_path, filename)
            try:
                with Image.open(file_path) as img:
                    images.append((file_path, img.convert('RGB')))
            except IOError:
                print(f"Error opening {file_path}. Skipping.")


    return images

def calculate_clip_scores(input_images: List[Tuple[str, Image.Image]], 
                          reference_images: List[Tuple[str, Image.Image]],
                          batch_size: int = 32) -> List[Tuple[str, float]]:
    # Load the CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    score_sums = defaultdict(float)
    score_counts = defaultdict(int)

    # Process images in batches
    for i in range(0, len(input_images), batch_size):
        input_batch = input_images[i:i+batch_size]
        input_images_batch = [img for _, img in input_batch]
        
        # Preprocess input images
        inputs = processor(images=input_images_batch, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            # Get input image features
            input_features = model.get_image_features(**inputs)
            
            # Calculate scores for each reference image
            for _, ref_img in reference_images:
                ref_inputs = processor(images=[ref_img], return_tensors="pt", padding=True)
                ref_features = model.get_image_features(**ref_inputs)
                
                # Calculate similarity scores
                similarity = 100 * torch.nn.functional.cosine_similarity(input_features, ref_features)
                scores = similarity.squeeze().tolist()
                
                # Handle single image case
                if isinstance(scores, float):
                    scores = [scores]
                
                # Accumulate scores
                for (input_path, _), score in zip(input_batch, scores):
                    score_sums[input_path] += score
                    score_counts[input_path] += 1

    # Calculate average scores
    avg_scores = [score_sums[path] / score_counts[path] for path in score_sums]
    
    return avg_scores


def calculate_clip_image_scores_folder(folder_path: str, reference_folder: str) -> float:

    input_images = load_images_from_folder(folder_path)
    reference_images = load_images_from_folder(reference_folder)

        # Check if the number of images is equal to 100
    if len(input_images) != 25:
        raise ValueError(f"The number of images in the folder must be exactly 25. Found {len(input_images)} images.")


    avg_scores = calculate_clip_scores(input_images, reference_images)

    return avg_scores