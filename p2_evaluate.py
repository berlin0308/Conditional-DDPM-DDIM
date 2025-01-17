import os
import numpy as np
from PIL import Image
from skimage.metrics import mean_squared_error
import argparse
import pathlib

def calculate_mse_for_folders(folder1, folder2):

    folder1_files = sorted([f for f in os.listdir(folder1) if f.endswith('.png')])
    folder2_files = sorted([f for f in os.listdir(folder2) if f.endswith('.png')])
    
    
    mse_list = []
    
    for file1, file2 in zip(folder1_files, folder2_files):

        img1 = np.array(Image.open(os.path.join(folder1, file1)))
        img2 = np.array(Image.open(os.path.join(folder2, file2)))
        
        if img1.shape != img2.shape:
            raise ValueError(f"Different Image Size: {file1} and {file2}")
        
        mse = mean_squared_error(img1, img2)
        mse_list.append((file1, mse))
    
    average_mse = np.mean([mse for _, mse in mse_list])
    
    return mse_list, average_mse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_image_folder', type=pathlib.Path, required=False, default='hw2_data/face/GT')
    parser.add_argument("--gen_image_folder", type=pathlib.Path, required=False, default='')
    args = parser.parse_args()

    mse_list, average_mse = calculate_mse_for_folders(args.gt_image_folder, args.gen_image_folder)
    print(f"MSE: {mse_list}")
    print(f"Avg. MSE: {average_mse}")
