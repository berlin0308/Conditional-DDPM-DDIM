import torch
import math
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from p2_model import DDIM
from UNet import UNet

class Slerp:
    def __init__(self, low, high):
        """Initialize with two image tensors"""
        self.low = low
        self.high = high

    def interpolate(self, alpha):
        """Perform spherical linear interpolation (slerp) on images"""
        # Flatten the images to 1D tensors for dot product
        low_flat = self.low.view(-1)  # flatten to 1D tensor
        high_flat = self.high.view(-1)  # flatten to 1D tensor
        
        # Compute the angle (omega) between the two vectors
        omega = torch.acos(torch.clamp(torch.dot(low_flat/low_flat.norm(), high_flat/high_flat.norm()), -1, 1))
        so = torch.sin(omega)
        
        # Handle the case when sin(omega) is 0 (i.e., vectors are too close)
        # if so == 0:
        #     return (1.0 - alpha) * self.low + alpha * self.high  # Linear interpolation in case of zero angle
        
        # Perform spherical linear interpolation and reshape back to original shape
        return (torch.sin((1.0 - alpha) * omega) / so) * self.low + (torch.sin(alpha * omega) / so) * self.high

class Lerp:
    def __init__(self, low, high):
        """Initialize with two image tensors"""
        self.low = low
        self.high = high

    def interpolate(self, alpha):
        """Perform linear interpolation (lerp) on images"""
        return (1.0 - alpha) * self.low + alpha * self.high


def load_image(image_path):
    """Loads an image and converts it to a tensor"""
    image = Image.open(image_path).convert('RGB')  # Load image and convert to RGB
    transform = transforms.ToTensor()  # Convert image to tensor
    return transform(image)

def load_noise(noise_path):
    """Loads a noise vector saved as .pt"""
    return torch.load(noise_path)

def gen_image(noise, model):
    
    channels = 3
    height = 256
    width = 256

    # Perform DDIM Sampling to generate image
    generated_image = model.sample(
        batch_size=1,
        channels=channels,
        height=height,
        width=width,
        ground_truth_noise=noise,  # Shape: (1, 3, 256, 256)
        save_intermediate=False
    )

    return generated_image

if __name__ == '__main__':

    SLERP = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load two noise vectors
    noise_00 = load_noise('hw2_data/face/noise/00.pt')
    noise_01 = load_noise('hw2_data/face/noise/01.pt')

    alphas = torch.linspace(0, 1, steps=11)  # α = {0.0, 0.1, 0.2, ..., 1.0}

    if SLERP:
        slerp = Slerp(noise_00, noise_01)
        interpolated_noises = [slerp.interpolate(alpha) for alpha in alphas]
    else:
        lerp = Lerp(noise_00, noise_01)
        interpolated_noises = [lerp.interpolate(alpha) for alpha in alphas]

    generated_images = []

    unet_model = UNet().to(device)
    state_dict = torch.load("hw2_data/face/UNet.pt", map_location=device)
    unet_model.load_state_dict(state_dict)
    unet_model.eval()


    DDIM_model = DDIM(
        model=unet_model,
        n_timesteps=1000,
        n_steps=50,
        eta=0.0,  # Set to 0 for deterministic sampling
        device=device
    )
    for noise in interpolated_noises:
        with torch.no_grad():
            output_image = gen_image(noise=noise, model=DDIM_model)
            generated_images.append(output_image)

    grid_image = make_grid(torch.cat(generated_images, dim=0), nrow=len(alphas), padding=2, normalize=True)
    save_image(grid_image, 'slerp_concat.png')
