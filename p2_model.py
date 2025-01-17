import torch
import numpy as np
from utils import beta_scheduler
from UNet import UNet  # Ensure UNet.py is in the same directory
from torchvision.utils import save_image
import os

class DDIM:
    def __init__(self, model, n_timesteps=1000, n_steps=50, eta=0.0, device='cuda'):
        """
        Initialize the DDIM sampler.

        Args:
            model (nn.Module): Pre-trained UNet model to predict epsilon.
            n_timesteps (int): Total number of timesteps (default: 1000).
            n_steps (int): Number of sampling steps (default: 50).
            eta (float): Noise scale parameter (0 for deterministic sampling).
            device (str): Device to perform computations on ('cuda' or 'cpu').
        """
        self.model = model
        self.n_timesteps = n_timesteps
        self.n_steps = n_steps
        self.eta = eta
        self.device = device

        # Load beta schedule from utils.py
        beta = beta_scheduler(n_timestep=self.n_timesteps).to(self.device).float()  # Convert to float32
        self.beta = beta
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alpha_cumprod[:-1]], dim=0)

        # Precompute square roots and other constants
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.sqrt_alpha_cumprod_prev = torch.sqrt(self.alpha_cumprod_prev)
        self.sqrt_one_minus_alpha_cumprod_prev = torch.sqrt(1.0 - self.alpha_cumprod_prev)

        # Create uniform timesteps
        self.timesteps = self.get_uniform_timesteps()

    def get_uniform_timesteps(self):
        # Calculate step size to get uniform steps
        step_size = self.n_timesteps // self.n_steps

        # Create the sequence of timesteps and add 1 to shift
        ddim_timestep_seq = np.asarray(list(range(0, self.n_timesteps, step_size))) + 1

        # Compute the previous timestep sequence explicitly
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])

        # Store both current and previous timestep sequences
        self.timesteps = ddim_timestep_seq
        self.prev_timesteps = ddim_timestep_prev_seq

        return ddim_timestep_seq


    def ddim_step(self, x_t, t, clip_denoised=True):
        # Find the correct index of t in the timestep sequence
        prev_t = self.prev_timesteps[np.where(self.timesteps == t)[0][0]]

        # Retrieve coefficients for timestep t and prev_t
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t - 1]
        sqrt_alpha_cumprod_prev = self.sqrt_alpha_cumprod_prev[prev_t]

        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t - 1]
        sqrt_one_minus_alpha_cumprod_prev = self.sqrt_one_minus_alpha_cumprod_prev[prev_t]

        # Predict epsilon using the model
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
        epsilon_theta = self.model(x_t, t_tensor)

        # Compute x0 prediction
        x0_pred = (x_t - sqrt_one_minus_alpha_cumprod_t * epsilon_theta) / sqrt_alpha_cumprod_t
        if clip_denoised:
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        # Compute sigmas_t using the shifted timesteps
        sigmas_t = self.eta * torch.sqrt(
            (1 - self.alpha_cumprod_prev[prev_t])
            / (1 - self.alpha_cumprod[t - 1])
            * (1 - self.alpha_cumprod[t - 1] / self.alpha_cumprod_prev[prev_t])
        )

        # Compute pred_dir_xt
        pred_dir_xt = torch.sqrt(1 - self.alpha_cumprod_prev[prev_t] - sigmas_t**2) * epsilon_theta

        if self.eta == 0:
            # Without random noise
            x_prev = torch.sqrt(self.alpha_cumprod_prev[prev_t]) * x0_pred + pred_dir_xt
        else:
            # Add noise for non-deterministic sampling
            x_prev = torch.sqrt(self.alpha_cumprod_prev[prev_t]) * x0_pred + pred_dir_xt
            x_prev += sigmas_t * torch.randn_like(x_prev)
            
        # # Compute x_prev
        # x_prev = torch.sqrt(self.alpha_cumprod_prev[prev_t]) * x0_pred + pred_dir_xt
        # x_prev += sigmas_t * torch.randn_like(x_prev)

        return x_prev



    def sample(self, batch_size, channels, height, width, ground_truth_noise=None, save_intermediate=False, save_dir='samples'):
        """
        Generate samples using the DDIM sampler.

        Args:
            batch_size (int): Number of samples to generate.
            channels (int): Number of image channels.
            height (int): Image height.
            width (int): Image width.
            ground_truth_noise (torch.Tensor, optional): Predefined noise tensor for deterministic sampling.
            save_intermediate (bool): Whether to save intermediate images.
            save_dir (str): Directory to save images.

        Returns:
            torch.Tensor: Generated images.
        """
        if save_intermediate:
            os.makedirs(save_dir, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            if ground_truth_noise is not None:
                x_t = ground_truth_noise.to(self.device)
            else:
                x_t = torch.randn(batch_size, channels, height, width, device=self.device)

            if save_intermediate:
                # Denormalize before saving
                x_t_denorm = (x_t + 1) / 2
                x_t_denorm = torch.clamp(x_t_denorm, 0, 1)
                save_image(x_t_denorm, os.path.join(save_dir, "step_0.png"))
                print(f"Saved step 0 as {os.path.join(save_dir, 'step_0.png')}")

            for i, t in enumerate(reversed(self.timesteps)):
                # x_t = self.ddim_step(x_t, t, clip_denoised=True)
                x_t = self.ddim_step(x_t, t, clip_denoised=False)

                if save_intermediate and (i + 1) % (self.n_steps // 5) == 0:
                    # Denormalize before saving
                    x_t_denorm = (x_t + 1) / 2
                    x_t_denorm = torch.clamp(x_t_denorm, 0, 1)
                    step_path = os.path.join(save_dir, f"step_{i + 1}.png")
                    save_image(x_t_denorm, step_path)
                    print(f"Saved step {i + 1} as {step_path}")

            return x_t
