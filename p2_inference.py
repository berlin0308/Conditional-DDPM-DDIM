import torch
from p2_model import DDIM
from UNet import UNet
from torchvision.utils import save_image
import os
import argparse
import pathlib

def gen_image(noise_path, model):
    
    channels = 3
    height = 256
    width = 256

    # Load the noise tensor
    ground_truth_noise = torch.load(noise_path, map_location=device) #, weights_only=True)
    print(f"Loaded noise from {noise_path}")

    # Ensure the noise tensor has the correct shape
    expected_shape = (1, channels, height, width)  # Shape: [1, 3, 256, 256]
    if ground_truth_noise.shape != expected_shape:
        print(f"Noise tensor shape {ground_truth_noise.shape} does not match expected {expected_shape}. Skipping.")

    # Perform DDIM Sampling to generate image
    generated_image = model.sample(
        batch_size=1,
        channels=channels,
        height=height,
        width=width,
        ground_truth_noise=ground_truth_noise,  # Shape: (1, 3, 256, 256)
        save_intermediate=False
    )

    return generated_image


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_folder', type=pathlib.Path, required=False, default='hw2_data/face/noise')
    parser.add_argument("--output_folder", type=pathlib.Path, required=False, default='ddim_output')
    parser.add_argument('--unet_model_path', type=pathlib.Path, required=False, default='hw2_data/face/UNet.pt')
    args = parser.parse_args()

    # noise_dir = "hw2_data/face/noise"

    eta = 0.0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    unet_model = UNet().to(device)
    state_dict = torch.load(args.unet_model_path, map_location=device) #, weights_only=True)

    unet_model.load_state_dict(state_dict)
    unet_model.eval()
    print(f"Loaded pre-trained UNet model from {args.unet_model_path}")

    DDIM_model = DDIM(
        model=unet_model,
        n_timesteps=1000,
        n_steps=50,
        eta=eta,  # Set to 0 for deterministic sampling
        device=device
    )


    # output_folder = f'output_v1_eta{eta}'
    os.makedirs(args.output_folder, exist_ok=True)
    for i in range(10):
        noise_path = os.path.join(args.noise_folder, f"{i:02d}.pt")
        if not os.path.exists(noise_path):
            print(f"Noise file {noise_path} does not exist. Skipping.")
            continue
    
        output_image = gen_image(noise_path=noise_path, model=DDIM_model)

        # Save the generated image
        output_image_path = os.path.join(args.output_folder, f"{i:02d}.png")

        min_val = torch.min(output_image)
        max_val = torch.max(output_image)

        # Min-Max Normalization
        norm_output_image = (output_image - min_val) / (max_val - min_val)

        save_image(norm_output_image, output_image_path)
        print(f"Saved output image as {output_image_path}\n")
