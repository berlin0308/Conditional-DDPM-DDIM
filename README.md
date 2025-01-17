[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/3NjwmDJu)
# DLCV-Fall-2024-HW2
Please click [this link](https://docs.google.com/presentation/d/1nWH_CmF6iba0kQmi0TV_yI2Emu1EvjrX/edit?usp=sharing&ouid=107585355306558125830&rtpof=true&sd=true) to view the slides of HW2

## Deadline

**2024/10/29 (Tue.) 23:59 (GMT+8)**

## Packages

This homework should be completed using Python 3.8.5. For a list of packages you are allowed to import in this assignment, please refer to the `stable-diffusion/environment.yaml` for more details.

### Installation

You can run the following commands to install all the packages listed in the `environment.yaml`:

```sh
cd stable-diffusion
conda env create -f environment.yaml
conda activate ldm
```

To install the stable-diffusion v1.4. You can download the sd-v1-4.ckpt file from [hugging face link](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/tree/main) or use wget

```sh
wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
```

place the model checkpoint in the following path:

```sh
stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt
```

You can run the following command to test if you successfully build the environment
```sh
python scripts/txt2img.py --prompt "a photograph of an astronaut riding a horse"
```

(Optional)
You can also install the environment with pip and requirements.txt, note that the version of pytorch should match your CUDA version. [pytorch CUDA version](https://pytorch.org/get-started/previous-versions/)
```sh
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
cd stable-diffusion
pip install -e .
```

> :warning: **Note**: Using packages with different versions will very likely lead to compatibility issues, so make sure that you install the correct version if one is specified above. E-mail or ask the TAs first if you want to import other packages.

> :warning: **Important**: You can **NOT** use **diffuser** for easy implementation!

## Hint

Your model is a DDPM, located in stable-diffusion/ldm/models/diffusion/ddpm.py. You may need to use model.cond_stage_model for the text encoder. Since Diffusion v1.4 is a latent diffusion model that operates in the latent space, functions such as model.encode_first_stage and model.get_first_stage_encoding may also be necessary. Additionally, model.q_sample could be useful for your process.

## Grading
We provide the code for evaluation of hw2_3
    
    python evaluation/grade_hw2_3.py --json_path hw2_data/textual_inversion/input.json --input_dir hw2_data/textual_inversion --output_dir output_folder_example 

## Directory Structure

The directory structure is organized as follows:
```
.
└── output folder/
    ├── 0/
    │   ├── 0/
    │   │   ├── source0_prompt0_0.png
    │   │   ├── ...
    │   │   └── source0_prompt0_25.png
    │   └── 1/
    │       ├── source0_prompt1_0.png
    │       ├── ...
    │       └── source0_prompt0_25.png
    └── 1/
        ├── 0/
        │   ├── source1_prompt0_0.png
        │   └── ...
        └── 1/
            ├── ...
            └── source1_prompt1_25.png
```

- **First Level**: Indices of concepts.
- **Second Level**: Indices of prompts.

In the private set, we will replace the `prompt` and `prompt_4_clip_eval` in `input.json`.

## Q&A

If you have any problems related to HW2, you may:

- **Use TA Hours**: Thu. 14:20~15:10 in MK514.
- **Contact TAs by Email**: [ntudlcv@gmail.com](mailto:ntudlcv@gmail.com) with the title `[DLCV hw2] Problems about ...`
- **Post Your Question**: Under the HW2 discussion section in NTU COOL.

Feel free to reach out if you have any questions or need further assistance!