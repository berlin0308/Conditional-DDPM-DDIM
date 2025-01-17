
# Conditional Diffusion Models: DDPM and DDIM

## Install Packages
With **Python3.8**, run the following command to install all the packages listed in the requirements.txt:

    pip3 install -r requirements.txt


## DDPM (Denoising Diffusion Probabilistic Models)

+ All DDPM implementation details are in ```p1_model.py```

+ A modified UNet that supports conditional learning by adding **time embedding** (time steps in the diffusion process) and **context embedding** (labels)

### Train DDPM on MNIST-M / SVHN Dataset

    python3 p1_train.py

### Generate Digit Images

Use the model you just trained **OR** download the pretrained model directly:

    gdown 1uX845gl81kgc-lHqo8SS78DBz8n1DWmS -O combined_ddpm.pth

Generate images for each digit (0-9) from MNIST-M / SVHN dataset:

    python3 p1_inference.py --output_image_dir <your_output_dir> --model_path <your_ddpm_model_path>

For example:

| MNIST-M | SVHN | 
| -------- | -------- |
| ![image](https://github.com/user-attachments/assets/b85d71a9-b650-4d5a-ac5b-bfba37b00f18) | ![image](https://github.com/user-attachments/assets/9798dff2-1e7a-4143-aeac-8fd8e5b7059f) |


### Evaluation

The output images can be evaluated by ```digit_classifier.py```

    python3 digit_classifier.py --folder <your_output_dir> --checkpoint <mnistm_ckpt_path>

|  | MNIST-M | SVHN | Average |
| -----| -------- | -------- | --- | 
| Acuracy |  99.80%  | 99.20%     | 99.50% |


## DDIM (Denoising Diffusion Implicit Models)

+ All DDPM implementation details are in ```p2_model.py```

### Generate Human Face Images

    python3 p2_inference.py --noise_folder <input_noise_dir> --output_folder <your_output_dir> --unet_model_path <your_unet_path>

With different ETA (Avg. MSE: 1.56):

![image](https://github.com/user-attachments/assets/b557fc0c-60e7-4099-8f81-8e1c858a7a9b)

### Interpolate Two Noises

    python3 p2_interpolate.py

+ Linear interpolation (LERP): Simply linearly mixes the two noise vectors without considering their position on a hypersphere (by setting ```SLERP=False```)
+ Spherical Linear Interpolation (SLERP): Along the surface of a hypersphere (by setting ```SLERP=True```)

| LERP | SLERP | 
| -------- | -------- |
| ![image](https://github.com/user-attachments/assets/ae226b9e-ef3e-4eb8-9934-48ea857bea8b) | ![image](https://github.com/user-attachments/assets/b3dcdce6-8ad5-4692-aabe-50f6dc5de5f8) |





