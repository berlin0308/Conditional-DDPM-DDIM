import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as trns
from torchvision.utils import save_image, make_grid
import os
from PIL import Image
import csv
import argparse
import pathlib
from p1_model import CombinedDDPM, DDPM, ContextUnet
from torch.cuda.amp import autocast


class ImageDataset(Dataset):
    def __init__(self, file_path, csv_path, transform=None):
        self.csv_path = csv_path
        self.path = file_path
        self.transform = transform
        if transform:
            self.transform = transform
        else:
            self.transform = trns.Compose(
                [
                    trns.Resize([32, 28]),
                    trns.ToTensor(),
                    trns.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        self.imgname_csv = []
        self.labels_csv = []
        self.files = []
        self.labels = []
        with open(self.csv_path, "r", newline="") as file:
            reader = csv.reader(file, delimiter=",")
            next(reader)
            for row in reader:
                img_name, label = row
                self.imgname_csv.append(img_name)
                self.labels_csv.append(torch.tensor(int(label)))

        for x in os.listdir(self.path):
            if x.endswith(".png") and x in self.imgname_csv:
                self.files.append(os.path.join(self.path, x))
                self.labels.append(self.labels_csv[self.imgname_csv.index(x)])

    def __getitem__(self, idx):
        data = Image.open(self.files[idx])
        data = self.transform(data)
        return data, self.labels[idx]

    def __len__(self):
        return len(self.files)


def output_images(save_dir, model_path):

    # hardcoding these here
    n_T = 500  # 500
    print(torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    n_classes = 10

    ddpm1 = DDPM(
        nn_model=ContextUnet(in_channels=3, n_feat=256, n_classes=10),
        betas=(1e-4, 0.02),
        n_T=500,
        device=device,
        drop_prob=0.0,
    )

    ddpm2 = DDPM(
        nn_model=ContextUnet(in_channels=3, n_feat=128, n_classes=10),
        betas=(1e-4, 0.02),
        n_T=500,
        device=device,
        drop_prob=0.0
    )

    ddpm = CombinedDDPM(ddpm1=ddpm1, ddpm2=ddpm2)
    ddpm.load_state_dict(torch.load(model_path, map_location=device))
    ddpm.to(device)

    # save_dir = "p1_sample_output/"

    """
    MNIST-M dataset
    """
    mnistm_images_path = os.path.join(save_dir, 'mnistm')
    os.makedirs(mnistm_images_path, exist_ok=True)
    # mnistm_model_path = "p1_mnistm_b512_ft256_lr1e-4/model_99.pth"


    # for eval, save an image of currently generated samples (top rows)
    # followed by real images (bottom rows)
    ddpm.eval()
    with torch.no_grad():
        n_sample = 50 * n_classes

        with torch.autocast(device_type=device):
            x_gen, x_store = ddpm.sample(n_sample, (3, 28, 28), device, guide_w=2, mode=1)

        # x_concat_list = []
        # x_column_list = []
        for i in range(n_sample):
            # x_column_list.append(x_gen[i])
            # if i % 10 == 9:
            #     column = torch.cat(x_column_list, dim=1)
            #     x_concat_list.append(column)
            #     x_column_list = []

            save_image(x_gen[i], os.path.join(mnistm_images_path, f"{i%10}_{int((i-i%10)/10)+1:03d}.png"))

        # x_concat = torch.cat(x_concat_list, dim=2)
        # save_image(x_concat, os.path.join(mnistm_images_path, "concat_img.png"))


    """
    SVHN dataset
    """
    svhn_images_path = os.path.join(save_dir, 'svhn')
    os.makedirs(svhn_images_path, exist_ok=True)
    # svhn_model_path = "p1_svhn_b256_f128_lr2e-4_d0.3/model_99.pth"


    # for eval, save an image of currently generated samples (top rows)
    # followed by real images (bottom rows)
    ddpm.eval()
    with torch.no_grad():
        n_sample = 50 * n_classes

        with torch.autocast(device_type=device):
            x_gen, x_store = ddpm.sample(n_sample, (3, 28, 28), device, guide_w=2, mode=2)

        # x_concat_list = []
        # x_column_list = []
        for i in range(n_sample):
            # x_column_list.append(x_gen[i])
            # if i % 10 == 9:
            #     column = torch.cat(x_column_list, dim=1)
            #     x_concat_list.append(column)
            #     x_column_list = []

            save_image(x_gen[i], os.path.join(svhn_images_path, f"{i%10}_{int((i-i%10)/10)+1:03d}.png"))

        # x_concat = torch.cat(x_concat_list, dim=2)
        # save_image(x_concat, os.path.join(svhn_images_path, "concat_img.png"))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_image_dir', type=pathlib.Path, required=True)
    parser.add_argument("--model_path", type=pathlib.Path, required=False, default='combined_ddpm.pth')
    args = parser.parse_args()
    
    os.makedirs(args.output_image_dir, exist_ok=True)

    """
    3-4 min on NVIDIA TITAN RTX (with torch.autocast(device_type=device))
    """
    from datetime import datetime
    print(datetime.now())
    output_images(save_dir=args.output_image_dir, model_path=args.model_path)
    print(datetime.now())